import scanpy as sc
import anndata as ad
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree
import random
import numpy as np
from random import choice
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import issparse
import torch



#######################################################################
## 2024.9.16 LLY add some function for Train and Test
#######################################################################
def extract_test_data(data_loader):
    reduced_expression_list = []
    reduced_image_list = []
    reduced_coord_list = []
    reduced_row_list = []
    reduced_col_list = []

    for batch in tqdm(data_loader):
        reduced_expression_list.append(batch['reduced_expression'])
        reduced_image_list.append(batch['image'])
        reduced_coord_list.append(batch['spatial_coords'])  
        reduced_row_list.append(batch['array_row'])  
        reduced_col_list.append(batch['array_col']) 

    print("***** batch_size=adata.shape[0] 不影响 *****")
    input_spot_test = torch.cat(reduced_expression_list)
    print(input_spot_test.shape)
    input_image_test = torch.cat(reduced_image_list)
    print(input_image_test.shape)

    input_coord_test = reduced_coord_list
    print(len(input_coord_test))
    input_row_test = reduced_row_list
    print(len(input_row_test))
    input_col_test = reduced_col_list
    print(len(input_col_test))
    print("***** *****")
    
    print("Finished extractting test data")    
    return input_spot_test, input_image_test, input_coord_test, input_row_test, input_col_test



#######################################################################
## 2024.9.16 LLY add some function for Load between spot data
#######################################################################






def loadBatchData(train_image_mat, train_matrix_mat, train_coors_mat, batch_size, pos_info):
    '''
    Generate batch training data   
    '''
    
    train_pos_dist = pos_info['pos dist']
    train_pos_ind = pos_info['pos ind']
    
    train_index_list = list(range(train_image_mat.shape[0]))
    random.shuffle(train_index_list)

    
    train_data_size = train_image_mat.shape[0]

    half_batch_size =  int(batch_size/2)
    batch_num = train_data_size//half_batch_size
    
    for i in range(batch_num):
        
        start = i*half_batch_size
        end = start + half_batch_size
        
        tmp_index_list =  list(train_index_list[start:end])
       
        pos_peer_index = []

        neighbor_index = np.zeros((batch_size, batch_size))
        
        count = 0
        pos_index_list = []
        for j in tmp_index_list:
             
            cur_pos_peer_index = np.copy(train_pos_ind[j])
            ## shummin           
            # random.shuffle(cur_pos_peer_index)
            # pos_index_list.append(cur_pos_peer_index[0])
            
            ## when only select itself, adjust this
            # random.shuffle(cur_pos_peer_index)
            pos_index_list.append(cur_pos_peer_index)
            
            neighbor_index[count][half_batch_size+count] = 1 
            neighbor_index[half_batch_size+count][count] = 1
 
            count += 1
     
        tmp_index_list.extend(pos_index_list)
        cur_index_list = np.asarray(tmp_index_list)
        cur_batch_mat = np.take(train_image_mat.cpu(), cur_index_list, axis=0)
        cur_matrix_mat = np.take(train_matrix_mat.cpu(), cur_index_list, axis=0)
        
        yield cur_batch_mat, cur_matrix_mat, neighbor_index, cur_index_list        

    pass













