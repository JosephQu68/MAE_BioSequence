import os
import numpy as np
import tensorflow as tf

def dataloader(file = 'scop_fa_represeq_lib_latest.fa', len_data = 5000, max_len_percintile=80):
    file_o = open(file, 'r')
    seq_list = []
    seq_length = []

    try:
        while True:
            temp_line = file_o.readline()
            if len(temp_line) == 0:
                break
            if temp_line[0] != '>':
                temp_line = temp_line.rstrip()
                temp_line = list(temp_line)
                seq_list.append(temp_line)
                #print(temp_line)
    finally:
        file_o.close()

    MAX_LENGTH = 0
    seq_list = seq_list[:len_data]

    for single_list in seq_list:
        seq_length.append(len(single_list))
    seq_length = np.array(seq_length)
    MAX_LENGTH = int(np.percentile(seq_length,max_len_percintile))
    
    return seq_list, MAX_LENGTH


#   得到词典
def get_dict(data):
    voc = set()
    for single_list in data:
        for single_chr in single_list:
            voc.add(single_chr)

    voc = list(voc)

    voc_char_to_int = dict()
    voc_int_to_char = dict()
    for i in range(len(voc)):
        voc_char_to_int[voc[i]] = i
    for i in range(len(voc)):
        voc_int_to_char[i] = voc[i]
    return voc_char_to_int, voc_int_to_char


def seq_data_to_onehot(seq_data, voc_char_to_int,max_length):
    onehot_list = list() #转化为integer的序列
    for seq in seq_data:
        # temp = _seq_to_integer(seq,voc_char_to_int)
        # # print(temp)
        # temp = _integer_to_onehot(temp,max_length,len(voc_char_to_int))
        # # print(temp)
        # onehot_list.append(temp)
        tmp_onehot = np.zeros((max_length,len(voc_char_to_int)),dtype=np.float32)
        i = 0
        for single_char in seq:
            tmp_onehot[i,voc_char_to_int[single_char]] = 1.
            i += 1
            if i >= max_length:
                break
        onehot_list.append(tmp_onehot)
    onehot_list = np.array(onehot_list, dtype=np.float32)
    return onehot_list



def onehot_to_seq(matrix_of_single_seq, voc_int_to_char):
# 将 onehot矩阵转化成单个序列
    matrix_of_single_seq = np.abs(matrix_of_single_seq)
    num_seq = np.argmax(matrix_of_single_seq, 1)
    res = ''
    for num in num_seq:
        res += voc_int_to_char[num]
    return res

def mask_onehot_matrix(onehot_data, mask_rate = 0.2):
    # To input the whole data
    len_data = onehot_data.shape[0]
    res = onehot_data.copy()

    val_len = np.sum(np.sum(onehot_data, axis=2),axis=1)
    mask_len = (val_len * mask_rate).astype(np.int32)
    right_limit = val_len-mask_len

    start_point = np.random.randint(right_limit)
    end_point = start_point+mask_len
    
    # len_seq = onehot_data.shape[1]
    # len_mask = int(mask_rate*len_seq)
    # for single_matrix in res:
    #     mask_choose = np.random.choice(len_seq,len_mask,replace=False)
    #     single_matrix[mask_choose,:]=0
    for _ in range(len_data):
        res[_,start_point[_]:end_point[_],:] = 0.
    return res

from MAESeqModule.MAESeq_model import ReconstructRateVaried
import pandas as pd

def evaluate_per_mask_rate(onehot_test, autoencoder):
    mask_rates = np.linspace(0,1,21)
    res = pd.Series(dtype=pd.Float64Dtype)
    for rate in mask_rates:
        onehot_test_mask = mask_onehot_matrix(onehot_test, rate)
        test_res = autoencoder.predict(onehot_test_mask)
        reconst_rate = ReconstructRateVaried(onehot_test, test_res)
        # reconst_rate = rate
        res['Mask '+'%.2f'%rate] = float(reconst_rate)
    return res

def extract_history(history):
    res_data_dict = {
            'Loss':history.history['loss'],
            'ValLoss':history.history['val_loss'],
            'ReconstructRate':history.history['ReconstructRateVaried'],
            'ValReconstructRate':history.history['val_ReconstructRateVaried']
        }
    res_data = pd.DataFrame(res_data_dict)
    return res_data
    

# def my_loss(y_true, y_prod,dict):
#     cnt_res = 0
#     nums_of_batch = y_true.shape[0]
#     len_seq = y_true.shape[1]
#     for i in range(nums_of_batch):
#       y_prod_temp = y_prod[i]
#       y_true_temp = y_true[i]
#       seq_pord = onehot_to_seq(y_prod_temp,dict)
#       seq_true = onehot_to_seq(y_true_temp,dict)
#       cnt = 0
#       for j in range(len(seq_pord)):
#           if seq_pord[j] == seq_true[j]:
#               cnt += 1
#       cnt_res += (cnt/len_seq)
#     return cnt_res / nums_of_batch
