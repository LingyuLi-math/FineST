import torch
import numpy as np
from .utils import *
from .loadData import * 


#################################################################
# 2024.9.16 NameError: name 'loadBatchData' is not defined
#################################################################
# def loadBatchData(train_image_mat, train_matrix_mat, train_coors_mat, batch_size, pos_info):
#     '''
#     Generate batch training data   
#     '''
    
#     train_pos_dist = pos_info['pos dist']
#     train_pos_ind = pos_info['pos ind']
    
#     train_index_list = list(range(train_image_mat.shape[0]))
#     random.shuffle(train_index_list)

    
#     train_data_size = train_image_mat.shape[0]

#     half_batch_size =  int(batch_size/2)
#     batch_num = train_data_size//half_batch_size
    
#     for i in range(batch_num):
        
#         start = i*half_batch_size
#         end = start + half_batch_size
        
#         tmp_index_list =  list(train_index_list[start:end])
       
#         pos_peer_index = []

#         neighbor_index = np.zeros((batch_size, batch_size))
        
#         count = 0
#         pos_index_list = []
#         for j in tmp_index_list:
             
#             cur_pos_peer_index = np.copy(train_pos_ind[j])
#             ## shummin           
#             # random.shuffle(cur_pos_peer_index)
#             # pos_index_list.append(cur_pos_peer_index[0])
            
#             ## when only select itself, adjust this
#             # random.shuffle(cur_pos_peer_index)
#             pos_index_list.append(cur_pos_peer_index)
            
#             neighbor_index[count][half_batch_size+count] = 1 
#             neighbor_index[half_batch_size+count][count] = 1
 
#             count += 1
     
#         tmp_index_list.extend(pos_index_list)
#         cur_index_list = np.asarray(tmp_index_list)
#         cur_batch_mat = np.take(train_image_mat.cpu(), cur_index_list, axis=0)
#         cur_matrix_mat = np.take(train_matrix_mat.cpu(), cur_index_list, axis=0)
        
#         yield cur_batch_mat, cur_matrix_mat, neighbor_index, cur_index_list        

#     pass





#################################################################
# 2024.9.16 NameError: name 'loadTrainTestData' is not defined
#################################################################
# def loadTrainTestData(train_loader, neighbor_k):
    
#     tqdm_object = tqdm(train_loader, total=len(train_loader))

#     matrix_data = []
#     image_data = []
#     spatial_coords_list = []
#     array_row_list = []
#     array_col_list = []

#     for batch in tqdm_object:
#         # Load data
#         matrix_data.append(batch["reduced_expression"].clone().detach().cuda())
#         image_data.append(batch["image"].clone().detach().cuda())

#         # Process each batch's spatial_coords
#         spatial_coords = batch["spatial_coords"]
#         combined_coords = torch.stack((spatial_coords[0], spatial_coords[1]), dim=1)
#         spatial_coords_list.append(combined_coords)

#         array_row = batch["array_row"]
#         array_row_list.append(array_row)
#         array_col = batch["array_col"]
#         array_col_list.append(array_col)

#     # Matrix data
#     matrix_tensor = torch.cat(matrix_data).to(device)
#     # Coord data
#     spatial_coords_list_all = torch.cat(spatial_coords_list).to(device)
#     array_row_list_all = torch.cat(array_row_list).to(device)
#     array_col_list_all = torch.cat(array_col_list).to(device)
#     # Image data
#     image_tensor = torch.cat(image_data).to(device)
#     image_tensor = image_tensor.view(image_tensor.shape[0] * image_tensor.shape[1], image_tensor.shape[2])
#     inputdata_reshaped, latent_image_reshape = reshape_latent_image(image_tensor)
#     latent_representation_image_arr = latent_image_reshape.cpu().detach().numpy()

#     # Create adata_latent object
#     adata_latent = anndata.AnnData(X=latent_representation_image_arr)
#     adata_latent.obsm['spatial'] = np.array(spatial_coords_list_all.cpu())
#     adata_latent.obs['array_row'] = np.array(array_row_list_all.cpu())
#     adata_latent.obs['array_col'] = np.array(array_col_list_all.cpu())

#     train_genes = adata_latent.var_names

#     # Generate training data representation, training coordinate matrix, and positive sample information
#     cur_train_data_mat = inputdata_reshaped
#     cur_train_coors_mat = np.column_stack((adata_latent.obs['array_row'], adata_latent.obs['array_col']))
#     cur_train_matrix_mat = matrix_tensor

#     # Generate positive pair information
#     pos_dist, pos_ind = checkNeighbors(adata_latent, neighbor_k)
#     cur_pos_info = {'pos dist': pos_dist, 'pos ind': pos_ind}

#     return cur_train_data_mat, cur_train_matrix_mat, cur_train_coors_mat, cur_pos_info


#################################################################
# 2024.9.16 NameError: name 'checkNeighbors' is not defined
#################################################################
# def checkNeighbors(cur_adata, neighbor_k):
#     '''
#     Return 'dist' and 'ind' of positive samples.    
#     '''
#     print("checkNeighbors.............")
    
#     cur_coor = np.column_stack((cur_adata.obs['array_row'].values, cur_adata.obs['array_col'].values))
#     cur_coor_tree = KDTree(cur_coor, leaf_size=2)
#     location_dist, location_ind  = cur_coor_tree.query(cur_coor, k=(neighbor_k+1))
#     ## Need to consider the selected location itself
#     location_dist = location_dist[:,0]
#     location_ind = location_ind[:,0]

#     ## shumin
#     # location_dist = location_dist[:,1:]
#     # location_ind = location_ind[:,1:]
    
#     return location_dist, location_ind



def adjust_learning_rate(optimizer, epoch, initial_lr, num_epochs):
    
    """Adjusts the learning rate based on the cosine annealing strategy."""
    
    lr = 0.5 * initial_lr * (1 + np.cos(np.pi * epoch / num_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_model(model, dir_name, params, optimizer, LOSS):
    cur_save_path = os.path.join(dir_name, "epoch_"+str(params["training_epoch"])+".pt")
    torch.save({
                  'epoch': params['training_epoch'],
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': LOSS,
                  'params': params,
                  # 'train_genes':train_genes,
                   }, cur_save_path)
    

# def reshape_latent_image(inputdata):    ## [adata.shape[0]*256, 384]  -->  [adata.shape[0], 384]
#     inputdata_reshaped = inputdata.view(int(inputdata.shape[0]/16), 16, inputdata.shape[1])  # [adata.shape[0], 256, 384]
#     average_inputdata_reshaped = torch.sum(inputdata_reshaped, dim=1) / inputdata_reshaped.size(1)
#     return inputdata_reshaped, average_inputdata_reshaped

    
def train_model(params, model, train_loader, optimizer, cur_epoch, l): 
    
    print("train model")    
    
    cur_lr = adjust_learning_rate(optimizer, cur_epoch, params['inital_learning_rate'], params['training_epoch'])
    total_loss, total_num = 0.0, 0.0

    ## load data
    (cur_train_data_mat, 
     cur_train_matrix_mat, 
     cur_train_coors_mat, 
     cur_pos_info) = loadTrainTestData(train_loader, neighbor_k=params['k_nearest_positives'])

    
    for image_profile, gene_profile, positive_index, _ in loadBatchData(cur_train_data_mat, 
                                                                        cur_train_matrix_mat,
                                                                        cur_train_coors_mat, 
                                                                        params['batch_size_pair'],
                                                                        cur_pos_info):
        
        input_gene_exp = torch.tensor(np.asarray(gene_profile)).float().to(device)   # torch.Size([64, 128])
        image_profile_reshape = image_profile.view(-1, image_profile.shape[2])     # [1331, 256, 384] --> [1331*256, 384]
        input_image_exp = image_profile_reshape.clone().detach().to(device)     # SDU
        
        ## model
        (representation_matrix, 
         reconstruction_matrix, 
         projection_matrix,
         representation_image, 
         reconstruction_iamge, 
         projection_image) = model(input_gene_exp, input_image_exp)     
        
        ## reshape
        _, representation_image_reshape = reshape_latent_image(representation_image)
        _, projection_image_reshape = reshape_latent_image(projection_image)
    
        ## cross decoder
        reconstructed_matrix_reshaped = model.matrix_decoder(representation_image)  
        _, reconstruction_iamge_reshapef2 = reshape_latent_image(reconstructed_matrix_reshaped)

        ## compute the loss
        loss = l(
            projection_image_reshape,
            projection_matrix,
            # representation_image_reshape,
            # representation_matrix,
            torch.tensor(positive_index).to(device),
            input_image_exp,
            reconstruction_iamge,
            reconstruction_matrix,
            reconstruction_iamge_reshapef2,
            # input_gene_exp,
            # w1=1, w2=1, w3=1, w4=1
            input_gene_exp
        )  

        # optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * gene_profile.shape[0]
        total_num += gene_profile.shape[0]
        pass
    
    LOSS = total_loss/total_num    # shumin end

    return LOSS



def test_model(params, model, test_loader, l): 
    
    print("test model")
    
    # load data
    (cur_train_data_mat, 
     cur_train_matrix_mat, 
     cur_train_coors_mat, 
     cur_pos_info) = loadTrainTestData(test_loader, neighbor_k=params['k_nearest_positives'])
        
    total_loss, total_num = 0.0, 0.0
    for image_profile, gene_profile, positive_index, _ in loadBatchData(cur_train_data_mat, 
                                                                        cur_train_matrix_mat,
                                                                        cur_train_coors_mat, 
                                                                        params['batch_size_pair'],
                                                                        cur_pos_info):
        
        input_gene_exp = torch.tensor(np.asarray(gene_profile)).float().to(device)   # torch.Size([64, 128])
        image_profile_reshape = image_profile.view(-1, image_profile.shape[2])     # [1331, 256, 384] --> [1331*256, 384]
        input_image_exp = image_profile_reshape.clone().detach().to(device)     # SDU
        
        ## model
        (representation_matrix, 
         reconstruction_matrix, 
         projection_matrix,
         representation_image, 
         reconstruction_iamge, 
         projection_image) = model(input_gene_exp, input_image_exp)     
        
        ## reshape
        _, representation_image_reshape = reshape_latent_image(representation_image)
        _, projection_image_reshape = reshape_latent_image(projection_image)
    
        ## cross decoder
        reconstructed_matrix_reshaped = model.matrix_decoder(representation_image)  
        _, reconstruction_iamge_reshapef2 = reshape_latent_image(reconstructed_matrix_reshaped)

        ## compute the loss
        loss = l(
            projection_image_reshape,
            projection_matrix,
            # representation_image_reshape,
            # representation_matrix,
            torch.tensor(positive_index).to(device),
            input_image_exp,
            reconstruction_iamge,
            reconstruction_matrix,
            reconstruction_iamge_reshapef2,
            # input_gene_exp,
            # w1=1, w2=1, w3=1, w4=1
            input_gene_exp
        )  
    
        total_loss += loss.item() * gene_profile.shape[0]
        total_num += gene_profile.shape[0]
        pass
    
    LOSS = total_loss/total_num 
    
    # count = batch["reduced_expression"].size(0)
    # loss_meter.update(loss.item(), count)
    # tqdm_object.set_postfix(valid_loss=loss_meter.avg)

    return LOSS



# def main():
    
#     parser = ArgumentParser(description="Train a cellContrast model")
    
#     parser.add_argument('--train_data_path', type=str,
#                         help="The path of training data with h5ad format (annData object)")
#     parser.add_argument('--save_folder', type=str,
#                         help="Save folder of model related files, default:'./cellContrast_models'",default="./cellContrast_models")
    
#     parser.add_argument('--parameter_file_path', type=str,
#                         help="Path of parameter settings, default:'./parameters.json'",default="./parameters.json")
    
    
#     args = parser.parse_args()
    
#     if len(sys.argv[1:]) == 0:
#         parser.print_help()
#         sys.exit(1)
    
#     # check parameters
#     if(not os.path.exists(args.train_data_path)):
#         print("train data not exists!")
#         sys.exit(1)
    
#     if(not os.path.exists(args.parameter_file_path)):
#         print("parameter file not exists!")
#         sys.exit(1)
    
#     if(not os.path.exists(args.save_folder)):
#         os.mkdir(args.save_folder)
    
    
#     train_model(args)
#     pass


# if __name__ == '__main__':
    
#     main()