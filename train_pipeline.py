#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import tensorflow as tf
from keras.backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = '6'

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.compat.v1.Session(config=config))


# In[3]:


from MAESeqModule.MAESeq_utils import dataloader,get_dict, seq_data_to_onehot

seq_list, max_len = dataloader(
    file='dataset/scop_fa_represeq_lib_latest.fa',
    len_data=35000, max_len_percintile=75)
max_len = 300 # 在这里统一一下
dict_char2int, dict_int2char = get_dict(seq_list)
onehot_data = seq_data_to_onehot(seq_list,dict_char2int,max_len)
dimension = len(dict_char2int)

print("Length of seq is {}".format(max_len))
print("Dimension is {}".format(dimension))


# In[11]:


import numpy as np
onehot_train, onehot_val,onehot_test =np.split(onehot_data,[
        int(len(onehot_data)*0.8), 
        int(len(onehot_data)*0.95)])
print(onehot_train.shape)
print(onehot_val.shape)
print(onehot_test.shape)


# In[5]:


import tensorflow as tf
lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = 1e-3,
    decay_steps = 438*2,
    decay_rate = 0.93,
    staircase=True
)


# In[6]:


from MAESeqModule.MAESeq_utils import mask_onehot_matrix
MASK_RATE = 0.1
onehot_train_mask = mask_onehot_matrix(onehot_train,MASK_RATE)
onehot_val_mask = mask_onehot_matrix(onehot_test,MASK_RATE)
onehot_test_mask = mask_onehot_matrix(onehot_test,MASK_RATE)


# In[ ]:


from keras import losses,optimizers
from  MAESeqModule.MAESeq_model import AutoencoderGRU,my_loss_entropy,ReconstructRateVaried
autoencoder = AutoencoderGRU(latent_dim=512, encoder_shapes=(max_len, dimension))
# autoencoder.compile(optimizer='adam', loss= losses.categorical_crossentropy )
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_scheduler), 
                    loss = my_loss_entropy, 
                    metrics = [ReconstructRateVaried])


# In[ ]:


BATCH_SIZE = 64
history = autoencoder.fit(onehot_train_mask, onehot_train,
                    epochs=100,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    validation_data=(onehot_test_mask, onehot_test)) 
autoencoder.save('trained_model/')


# In[ ]:


import pandas as pd
res_data_dict = {
        'Loss':history.history['loss'],
        'ValLoss':history.history['val_loss'],
        'ReconstructRate':history.history['ReconstructRateVaried'],
        'ValReconstructRate':history.history['val_ReconstructRateVaried']
    }
res_data = pd.DataFrame(res_data_dict)
res_data.to_csv('res/res_data.csv')

loss_res = history.history['loss']
val_loss_res = history.history['val_loss']
reconstructRate = history.history['ReconstructRateVaried']
reconstructRateVal = history.history['val_ReconstructRateVaried']


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(loss_res,'r', label='loss')
plt.plot(val_loss_res, label = 'val_loss')
plt.legend()
plt.savefig('res/res_loss.jpg')


# In[ ]:


plt.plot(reconstructRate,'r', label='reconstructRate')
plt.plot(reconstructRateVal, label = 'val_reconstructRate')
plt.legend()
plt.savefig('res/res_reconstruct_rate.jpg')


# In[ ]:


test = autoencoder.predict(onehot_test_mask)


# In[ ]:


from MAESeqModule.MAESeq_utils import onehot_to_seq
print(ReconstructRateVaried(test, onehot_test,dict_int2char))
for i in range(10):
  print('原先')
  print(onehot_to_seq(onehot_test[i],dict_int2char))
  print('预测')
  print(onehot_to_seq(test[i],dict_int2char))

