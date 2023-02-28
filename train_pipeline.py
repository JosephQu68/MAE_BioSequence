#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


# In[ ]:


from MAESeqModule.MAESeq_utils import dataloader,get_dict, seq_data_to_onehot
import numpy as np
DICT_EXIST = True
DICT_PATH_CHAR2INT = 'dict/char2int.npy'
DICT_PATH_INT2CHAR = 'dict/int2char.npy'
DATA_PATH = 'dataset/scop_fa_represeq_lib_latest.fa'

seq_list, max_len = dataloader(
    file= DATA_PATH,
    len_data=35000, max_len_percintile=75)
max_len = 300 # 在这里统一一下

if DICT_EXIST:
    dict_char2int = np.load(DICT_PATH_CHAR2INT, allow_pickle = True).item()
    dict_int2char = np.load(DICT_PATH_INT2CHAR,allow_pickle = True).item()
else:
    dict_char2int, dict_int2char = get_dict(seq_list)
    np.save(DICT_PATH_CHAR2INT, dict_char2int)
    np.save(DICT_PATH_INT2CHAR, dict_int2char)

onehot_data = seq_data_to_onehot(seq_list,dict_char2int,max_len)
dimension = len(dict_char2int)

print("Length of seq is {}".format(max_len))
print("Dimension is {}".format(dimension))


# In[ ]:


import numpy as np
onehot_train, onehot_val,onehot_test =np.split(onehot_data,[
        int(len(onehot_data)*0.8), 
        int(len(onehot_data)*0.95)])
print(onehot_train.shape)
print(onehot_val.shape)
print(onehot_test.shape)


# In[ ]:


import tensorflow as tf
lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = 1e-3,
    decay_steps = 438*2,
    decay_rate = 0.93,
    staircase=True
)


# In[ ]:


# from MAESeqModule.MAESeq_utils import mask_onehot_matrix
# MASK_RATE = 0.1
# onehot_train_mask = mask_onehot_matrix(onehot_train,MASK_RATE)
# onehot_val_mask = mask_onehot_matrix(onehot_test,MASK_RATE)
# onehot_test_mask = mask_onehot_matrix(onehot_test,MASK_RATE)


# In[ ]:


from keras import losses,optimizers,backend
from  MAESeqModule.MAESeq_model  import AutoencoderGRU,my_loss_entropy,ReconstructRateVaried
from MAESeqModule.MAESeq_utils import mask_onehot_matrix,evaluate_per_mask_rate,extract_history

def isSaved(mask_rate):
    return False
import pandas as pd


train_mask_rates = np.linspace(0,1,11)[:-1]
dict_for_all = dict()
for MASK_RATE in train_mask_rates:
    backend.clear_session()
    onehot_train_mask = mask_onehot_matrix(onehot_train,MASK_RATE)
    onehot_val_mask = mask_onehot_matrix(onehot_test,MASK_RATE)
    onehot_test_mask = mask_onehot_matrix(onehot_test,MASK_RATE)
    
    autoencoder = AutoencoderGRU(latent_dim=512, encoder_shapes=(max_len, dimension))
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_scheduler), 
                    loss = my_loss_entropy, 
                    metrics = [ReconstructRateVaried])
    
    BATCH_SIZE = 64
    history = autoencoder.fit(onehot_train_mask, onehot_train,
                    epochs=100,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    validation_data=(onehot_val_mask, onehot_val))
                    
    if isSaved(MASK_RATE):
        autoencoder.save('trained_model/')
    res_train = extract_history(history)
    res_train.to_csv('res/train_history_mask_rate_%.2f_.csv'%MASK_RATE)
    evaluated = evaluate_per_mask_rate(onehot_test, autoencoder)
    evaluated.to_csv('res/eval_per_mask_rate_in_%.2f_.csv'%MASK_RATE)

    dict_for_all['Train_Mask_Rate = %.2f'%MASK_RATE] = evaluated
    

overall_res_per_mask_rate = pd.DataFrame(dict_for_all)
overall_res_per_mask_rate.to_csv('res/overall.csv')


# # In[ ]:


# loss_res = history.history['loss']
# val_loss_res = history.history['val_loss']
# reconstructRate = history.history['ReconstructRateVaried']
# reconstructRateVal = history.history['val_ReconstructRateVaried']


# # In[ ]:


# import matplotlib.pyplot as plt
# plt.plot(loss_res,'r', label='loss')
# plt.plot(val_loss_res, label = 'val_loss')
# plt.legend()
# plt.savefig('res/res_loss.jpg')


# # In[ ]:


# plt.plot(reconstructRate,'r', label='reconstructRate')
# plt.plot(reconstructRateVal, label = 'val_reconstructRate')
# plt.legend()
# plt.savefig('res/res_reconstruct_rate.jpg')


# # In[ ]:


# test = autoencoder.predict(onehot_test_mask)


# # In[ ]:


from MAESeqModule.MAESeq_utils import onehot_to_seq
print(ReconstructRateVaried(test, onehot_test))
for i in range(10):
  print('原先')
  print(onehot_to_seq(onehot_test[i],dict_int2char))
  print('预测')
  print(onehot_to_seq(test[i],dict_int2char))

