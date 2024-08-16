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


# def calLocationDistance(train_data,test_data):
    
    
#     train_coor = np.column_stack((train_data.obs['x'].values,train_data.obs['y'].values))
#     test_coor = np.column_stack((test_data.obs['x'].values,test_data.obs['y'].values))
#     train_coor_tree = KDTree(train_coor,leaf_size=2)
#     dist,ind  = train_coor_tree.query(test_coor,k=train_coor.shape[0])
    
    
#     return dist,ind


# def checkNeighbors(cur_adata,neighbor_k):
    
#     '''
#     return dist,ind of positive samples.    
#     '''
#     print("checkNeighbors.............")
    
#     cur_coor = np.column_stack((cur_adata.obs['x'].values,cur_adata.obs['y'].values))
#     cur_coor_tree = KDTree(cur_coor,leaf_size=2)
#     location_dist,location_ind  = cur_coor_tree.query(cur_coor,k=(neighbor_k+1))
#     location_dist = location_dist[:,1:]
#     location_ind = location_ind[:,1:]
    
#     return location_dist,location_ind


def checkNeighbors(cur_adata, neighbor_k):
    '''
    Return 'dist' and 'ind' of positive samples.    
    '''
    print("checkNeighbors.............")
    
    cur_coor = np.column_stack((cur_adata.obs['array_row'].values, cur_adata.obs['array_col'].values))
    cur_coor_tree = KDTree(cur_coor, leaf_size=2)
    location_dist, location_ind  = cur_coor_tree.query(cur_coor, k=(neighbor_k+1))
    ## Need to consider the selected location itself
    location_dist = location_dist[:,0]
    location_ind = location_ind[:,0]

    ## shumin
    # location_dist = location_dist[:,1:]
    # location_ind = location_ind[:,1:]
    
    return location_dist, location_ind


# def loadTrainData(adata, neighbor_k, sample_ID_name='embryo'):
    
#     '''
#     '''
    
#     if(sample_ID_name in adata.obs):
#         unique_samples = adata.obs[sample_ID_name].unique()
    
    
#     train_sample_number = unique_samples.shape[0]
    
#     train_rep = []
#     train_coors_mat = []
#     # a list of dictionary
#     pos_info = []
    
    
#     # generate training dataset
#     for cur_sample in unique_samples:
        
    
#         # Filter the cells corresponding to the current embryo
#         cur_sample_adata = adata[adata.obs[sample_ID_name] == cur_sample]
        
#         # check if the input is sparse matrix
#         if(issparse(cur_sample_adata.X)):
#             train_input =  np.asarray(cur_sample_adata.X.todense())
#         else:
#             train_input = cur_sample_adata.X

#         cur_train_rep = np.nan_to_num(train_input)
        
#         # generate positive pair information
#         pos_dist, pos_ind = checkNeighbors(cur_sample_adata,neighbor_k)
#         cur_pos_info = {'pos dist':pos_dist,'pos ind':pos_ind}
        
#         train_rep.append(cur_train_rep)
#         pos_info.append(cur_pos_info)
#         cur_train_coors_mat = np.column_stack((cur_sample_adata.obs['x'],cur_sample_adata.obs['y']))
#         train_coors_mat.append(cur_train_coors_mat)
    
    
#     return train_rep, train_coors_mat, pos_info


def loadTrainTestData(train_loader, neighbor_k):
    
    tqdm_object = tqdm(train_loader, total=len(train_loader))

    matrix_data = []
    image_data = []
    spatial_coords_list = []
    array_row_list = []
    array_col_list = []

    for batch in tqdm_object:
        # Load data
        matrix_data.append(batch["reduced_expression"].clone().detach().cuda())
        image_data.append(batch["image"].clone().detach().cuda())

        # Process each batch's spatial_coords
        spatial_coords = batch["spatial_coords"]
        combined_coords = torch.stack((spatial_coords[0], spatial_coords[1]), dim=1)
        spatial_coords_list.append(combined_coords)

        array_row = batch["array_row"]
        array_row_list.append(array_row)
        array_col = batch["array_col"]
        array_col_list.append(array_col)

    # Matrix data
    matrix_tensor = torch.cat(matrix_data).to(device)
    # Coord data
    spatial_coords_list_all = torch.cat(spatial_coords_list).to(device)
    array_row_list_all = torch.cat(array_row_list).to(device)
    array_col_list_all = torch.cat(array_col_list).to(device)
    # Image data
    image_tensor = torch.cat(image_data).to(device)
    image_tensor = image_tensor.view(image_tensor.shape[0] * image_tensor.shape[1], image_tensor.shape[2])
    inputdata_reshaped, latent_image_reshape = reshape_latent_image(image_tensor)
    latent_representation_image_arr = latent_image_reshape.cpu().detach().numpy()

    # Create adata_latent object
    adata_latent = anndata.AnnData(X=latent_representation_image_arr)
    adata_latent.obsm['spatial'] = np.array(spatial_coords_list_all.cpu())
    adata_latent.obs['array_row'] = np.array(array_row_list_all.cpu())
    adata_latent.obs['array_col'] = np.array(array_col_list_all.cpu())

    train_genes = adata_latent.var_names

    # Generate training data representation, training coordinate matrix, and positive sample information
    cur_train_data_mat = inputdata_reshaped
    cur_train_coors_mat = np.column_stack((adata_latent.obs['array_row'], adata_latent.obs['array_col']))
    cur_train_matrix_mat = matrix_tensor

    # Generate positive pair information
    pos_dist, pos_ind = checkNeighbors(adata_latent, neighbor_k)
    cur_pos_info = {'pos dist': pos_dist, 'pos ind': pos_ind}

    return cur_train_data_mat, cur_train_matrix_mat, cur_train_coors_mat, cur_pos_info


# def loadBatchData(train_data_mat,train_coors_mat,batch_size,pos_info):
#     '''
    
#     generate batch training data
    
#     '''
    
#     train_pos_dist = pos_info['pos dist']
#     train_pos_ind = pos_info['pos ind']
    
#     train_index_list = list(range(train_data_mat.shape[0]))
#     random.shuffle(train_index_list)
#     train_data_size = train_data_mat.shape[0]

#     half_batch_size =  int(batch_size/2)
#     batch_num = train_data_size//half_batch_size
    
#     for i in range(batch_num):
        
#         start = i*half_batch_size
#         end = start + half_batch_size
        
#         tmp_index_list =  list(train_index_list[start:end])
       
#         pos_peer_index = []

#         neighbor_index = np.zeros((batch_size,batch_size))
        
#         count = 0
#         pos_index_list = []
#         for j in tmp_index_list:
             
#              cur_pos_peer_index = np.copy(train_pos_ind[j])
#              random.shuffle(cur_pos_peer_index)
             
#              pos_index_list.append(cur_pos_peer_index[0])
             
#              neighbor_index[count][half_batch_size+count] = 1 
#              neighbor_index[half_batch_size+count][count] = 1
 
#              count += 1
     

#         tmp_index_list.extend(pos_index_list)
#         cur_index_list = np.asarray(tmp_index_list)
#         cur_batch_mat = np.take(train_data_mat,cur_index_list,axis=0)
#         cur_coor_mat = np.take(train_coors_mat,cur_index_list,axis=0)
        
        
#         yield cur_batch_mat,neighbor_index,cur_coor_mat,cur_index_list
#     pass


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













