#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


# In[ ]:


from tensorflow.keras import mixed_precision
import keras
SEED = 42
tf.keras.utils.set_random_seed(SEED)
mixed_precision.set_global_policy('mixed_float16')
tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True) # Enable XLA.


# Some hyperparameters

# In[ ]:


from MAESeqModule.MAESeq_utils import dataloader,get_dict, seq_data_to_onehot
import numpy as np
DICT_EXIST = True
DATASET_EXIST = False
DICT_PATH_CHAR2INT = 'dict/char2int.npy'
DICT_PATH_INT2CHAR = 'dict/int2char.npy'
DATA_PATH = 'dataset/scop_fa_represeq_lib_latest.fa'
DATA_NPY_PATH = '/geniusland/home/qufuchuan/dataset/uniref90_min20_max300_num120k.npy'

BATCH_SIZE = 256
EPOCHS = 500
LEARNING_RATE = 1e-4


if DATASET_EXIST:
    seq_list = np.load(DATA_NPY_PATH)
else:
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


# Data Slide

# In[ ]:


import numpy as np
onehot_train, onehot_val,onehot_test =np.split(onehot_data,[
        int(len(onehot_data)*0.8), 
        int(len(onehot_data)*0.99)])
print(onehot_train.shape)
print(onehot_val.shape)
print(onehot_test.shape)


# Masking

# In[ ]:


from MAESeqModule.MAESeq_utils import mask_onehot_matrix,evaluate_per_mask_rate,extract_history,mask_onehot_matrix_discrete

# onehot_train_mask = mask_onehot_matrix(onehot_train,0.15)
# onehot_val_mask = mask_onehot_matrix(onehot_val,0.15)

onehot_train_mask = mask_onehot_matrix_discrete(onehot_train,0.15)
onehot_val_mask = mask_onehot_matrix_discrete(onehot_val,0.15)
# train_dataset = tf.data.Dataset.from_tensor_slices((onehot_train_mask, onehot_train))
# train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

# val_dataset = tf.data.Dataset.from_tensor_slices((onehot_val_mask, onehot_val))
# val_dataset = val_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)


# In[ ]:


import tensorflow as tf
from MAESeqModule.MAESeq_scheduler import WarmUpCosine

total_steps = int((len(onehot_train) / BATCH_SIZE) * EPOCHS)
warmup_epoch_percentage = 0.10
warmup_steps = int(total_steps * warmup_epoch_percentage)
warmup_cosine_lr_scheduler = WarmUpCosine(
    learning_rate_base=LEARNING_RATE,
    total_steps=total_steps,
    warmup_learning_rate=0.0,
    warmup_steps=warmup_steps,
)

lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = 1e-3,
    decay_steps = int(2 * len(onehot_train)/BATCH_SIZE),
    decay_rate = 0.96,
    staircase=True
)


# In[ ]:


from keras import losses,optimizers,backend
from  MAESeqModule.MAESeq_model  import AutoencoderGRU_withMaskLoss,my_loss_entropy,ReconstructRateVaried
from MAESeqModule.MAESeq_utils import mask_onehot_matrix,evaluate_per_mask_rate,extract_history


autoencoder = AutoencoderGRU_withMaskLoss(latent_dim=64, encoder_shapes=(max_len, dimension),
                                          gru_layer_shape=[64,128,256])

MP_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
# MP_optimizer = tf.keras.optimizers.Adam(learning_rate=warmup_cosine_lr_scheduler)
MP_optimizer = mixed_precision.LossScaleOptimizer(MP_optimizer)
          
autoencoder.compile(optimizer=MP_optimizer, 
                    loss = my_loss_entropy, 
                    metrics = [ReconstructRateVaried])
    
# history = autoencoder.fit(onehot_train_mask,onehot_train,
#                     epochs=EPOCHS,
#                     batch_size=BATCH_SIZE,
#                     shuffle=True,
#                     validation_data= [onehot_train_mask,onehot_train],
#                     callbacks=tf.keras.callbacks.ReduceLROnPlateau())

history = autoencoder.fit(onehot_train_mask,onehot_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    validation_data= [onehot_val_mask,onehot_val])

hist_data = extract_history(history)
hist_data.to_csv('training_history_30k_bidir_discrete_masking_1e-4_small.csv')

