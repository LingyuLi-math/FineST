import scanpy as sc
# from cellContrast.model import *
# from cellContrast import utils
from argparse import ArgumentParser, SUPPRESS
import os
import json
import sys
from scipy import sparse
from sklearn.manifold import MDS
import numpy as np
import  logging
logging.getLogger().setLevel(logging.INFO)
from .utils import *



# set the device
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


# def load_model(args,params):
    
#     save_folder = os.path.join(args.model_folder,"epoch_"+str(params["training_epoch"])+".pt")
#     if(dev=="cpu"):
#         checkpoint = torch.load(save_folder,map_location="cpu")
#     else:
#         checkpoint = torch.load(save_folder)
    
#     model_state_dict = checkpoint['model_state_dict']
#     train_genes = checkpoint["train_genes"]
    
#     # load parameter settings
#     with open(args.parameter_file_path,"r") as json_file:
#         params = json.load(json_file)
#     params["n_input"] = len(train_genes)
   
#     # init the model
#     model = CellContrastModel(n_input=params['n_input'],\
#                               n_encoder_hidden=params["n_encoder_hidden"],n_encoder_latent=params["n_encoder_latent"],\
#                               n_encoder_layers=params["n_encoder_layers"],n_projection_hidden=params["n_projection_hidden"],\
#                               n_projection_output=params["n_projection_output"])
    
#     # load model states
#     model.load_state_dict(model_state_dict)    
#     return model,train_genes


def load_model(params):    
    save_folder = os.path.join(dir_name, "epoch_"+str(params["training_epoch"])+".pt")
    if(dev=="cpu"):
        checkpoint = torch.load(save_folder,map_location="cpu")
    else:
        checkpoint = torch.load(save_folder)
    
    model_state_dict = checkpoint['model_state_dict']
    
    # load parameter settings
    with open(parameter_file_path,"r") as json_file:
        params = json.load(json_file)
    params['n_input_matrix'] = len(gene_hv)
    params['n_input_image'] = 384
    
    # init the model
    model = CellContrastModel(n_input_matrix=params['n_input_matrix'],
                              n_input_image=params['n_input_image'],
                              n_encoder_hidden_matrix=params["n_encoder_hidden_matrix"],
                              n_encoder_hidden_image=params["n_encoder_hidden_image"],
                              n_encoder_latent=params["n_encoder_latent"],
                              n_projection_hidden=params["n_projection_hidden"],
                              n_projection_output=params["n_projection_output"],
                              n_encoder_layers=params["n_encoder_layers"]).to(device)    
    # load model states
    model.load_state_dict(model_state_dict)    
    return model




# def infer_representations(model,query_adata):
    
    
#     print("device",device)
#     if(sparse.issparse(query_adata.X)):
#         input_gene_exp = np.asarray(query_adata.X.todense())
#     else:
#         input_gene_exp = query_adata.X
        
#     torch.tensor(input_gene_exp).float().to(device)
#     model.eval()
#     representation, projection = model(torch.tensor(input_gene_exp).float().to(device))
#     if(dev!="cpu"):
#         representation = representation.cpu()
        
#     return representation.detach().numpy()

# def format_query(query_adata,train_genes):
    
#     tmp = set(train_genes) - set(query_adata.var_names)
#     if(len(tmp)>0):
#         sys.exit("[ERROR] train genes %s not found in the query data." % (" ".join(tmp)))
    
#     gene_indices = [query_adata.var_names.get_loc(gene) for gene in train_genes]
#     formatted_query_adata = query_adata[:, gene_indices]
    
#     return formatted_query_adata


# def runMDS(query_adata):
    
#     mds = MDS(n_components=2, dissimilarity='precomputed',n_jobs=-1)
#     dissimilarity_mat = 1 - query_adata.uns["cosine sim of rep"]
#     logging.info("Running MDS algorithm...")
#     embedded_points = mds.fit_transform(dissimilarity_mat)
#     logging.info("Running MDS algorithm DONE!")
#     query_adata.uns['de novo x'] = embedded_points[:,0]
#     query_adata.uns['de novo y'] = embedded_points[:,1]
    

    
# def map_to_ST(query_rep,ref_rep,ref_coors):
    
#     sorted_similarity_matrix,index_matrix,similarity_matrix =  utils.getModelSimMat(query_rep,ref_rep)
    
#     predicted_coors = []
    
#     for i, ind in enumerate(index_matrix):
        
#         # ith query data
#         # ind: index of reference data
#         cur_predict_st = ind[0]
#         cur_st_coor = ref_coors[cur_predict_st]
#         predicted_coors.append(cur_st_coor)
    
#     predicted_coors = np.asarray(predicted_coors)
    
#     return predicted_coors
    

# def perform_inference(query_adata,ref_adata,model,enable_denovo):
    
#     # inference
    
    
#     ## generate the representations of query
#     query_rep = infer_representations(model,query_adata)
    
#     ## generate SC-SC pairwise similarities
#     sorted_similarity_matrix,index_matrix,similarity_matrix = utils.getModelSimMat(query_rep,query_rep)
    
#     query_adata.uns["representation"] = query_rep
#     query_adata.uns["cosine sim of rep"] = similarity_matrix
    
#     ## generate de novo coordinates based on MDS algorithm
#     if(enable_denovo):
#         runMDS(query_adata)
    
        
    
    
#     # generate the representations of reference data
#     ref_rep = infer_representations(model,ref_adata)
#     if(('sptial' not in ref_adata.obs)):
#         if('x' in ref_adata.obs and 'y' in ref_adata.obs):
#             ref_coors = np.column_stack((ref_adata.obs['x'].values,ref_adata.obs['y'].values))
#         else:
#             sys.exit("[ERROR] No spatial info found in ST data")
#     else:
#         ref_coors = ref_adata['spatial'].values
            
#     # map query to reference locations
#     query_referenced_coordinates = map_to_ST(query_rep,ref_rep,ref_coors)
#     query_adata.uns["referenced x"] = query_referenced_coordinates[:,0]
#     query_adata.uns["referenced y"] = query_referenced_coordinates[:,1]
    
    
#     return query_adata




def perform_inference_image(model, test_loader):
    print("device",device)    

    #####################################################################################
    # for whole dataset
    #####################################################################################        
    print("***** Begin perform_inference: ******")
    
    input_spot_all, input_image_all, input_coord_all, _, _ = extract_test_data(test_loader)
            
    ## input image and matrix
    matrix_profile = input_spot_all.to(device)
    image_profile = input_image_all.to(device)
    ## reshape image
    image_profile_reshape = image_profile.view(-1, image_profile.shape[2])     # [1331, 256, 384] --> [1331*256, 384]
    input_image_exp = image_profile_reshape.clone().detach().to(device)     # SDU
    
    ## useful model
    representation_matrix = model.matrix_encoder(matrix_profile)
    reconstructed_matrix = model.matrix_decoder(representation_matrix)
    projection_matrix = model.matrix_projection(representation_matrix)  
    representation_image = model.image_encoder(input_image_exp) 
    reconstruction_iamge = model.image_decoder(representation_image)
    projection_image = model.image_projection(representation_image)

    ## reshape
    _, representation_image_reshape = reshape_latent_image(representation_image)
    _, projection_image_reshape = reshape_latent_image(projection_image)

    ## cross decoder
    reconstructed_matrix_reshaped = model.matrix_decoder(representation_image)  
    _, reconstruction_iamge_reshapef2 = reshape_latent_image(reconstructed_matrix_reshaped)
    
    
    #####################################################################################  
    # convert
    #####################################################################################  
    ## matrix
    matrix_profile = matrix_profile.cpu().detach().numpy() 
    reconstructed_matrix = reconstructed_matrix.cpu().detach().numpy() 
    reconstruction_iamge_reshapef2 = reconstruction_iamge_reshapef2.cpu().detach().numpy() 
    ## latent space
    representation_image_reshape = representation_image_reshape.cpu().detach().numpy() 
    representation_matrix = representation_matrix.cpu().detach().numpy() 
    ## latent space -- peojection
    projection_image_reshape = projection_image_reshape.cpu().detach().numpy() 
    projection_matrix = projection_matrix.cpu().detach().numpy() 
    ## image
    input_image_exp = input_image_exp.cpu().detach().numpy() 
    reconstruction_iamge = reconstruction_iamge.cpu().detach().numpy()
    # ## tensor recon_f2
    # reconstructed_matrix_reshaped = reconstructed_matrix_reshaped.cpu().detach().numpy()
    
    return (matrix_profile, 
            reconstructed_matrix, 
            reconstruction_iamge_reshapef2, 
            representation_image_reshape,
            representation_matrix,
            projection_image_reshape,
            projection_matrix,
            input_image_exp,
            reconstruction_iamge,
            reconstructed_matrix_reshaped,
            input_coord_all)
    

# def main():
    
    
#     parser = ArgumentParser(description="Inference with cellContrast model")
    
#     parser.add_argument('--query_data_path', type=str,
#                         help="The path of querying data with h5ad format (annData object)")
#     parser.add_argument('--model_folder', type=str,
#                         help="Save folder of model related files, default:'./cellContrast_models'",default="./cellContrast_models")
#     parser.add_argument('--parameter_file_path', type=str,
#                         help="Path of parameter settings, default:'./parameters.json'",default="./parameters.json")
#     parser.add_argument('--ref_data_path',type=str, help="reference ST data, used in generating the coordinates of SC data as the reference, usually should be the training data of the model")
    
#     # whether to enable de novo coordinates inference
#     parser.add_argument('--enable_denovo', action="store_true",help="(Optional) generate the coordinates de novo by MDS algorithm",default=False)
#     parser.add_argument('--save_path',type=str,help="Save path of the spatial reconstructed SC data",default="./reconstructed_sc.h5ad")
    
    
#     args = parser.parse_args()
    
#     # load params
#     with open(args.parameter_file_path,"r") as json_file:
#         params = json.load(json_file)
    
#     # load models
#     model, train_genes = load_model(args,params)
#     model.to(device)
#     print("model",model)
   
#     query_adata = sc.read_h5ad(args.query_data_path)
#     ref_adata =  sc.read_h5ad(args.ref_data_path)
    
#     ## check if the train genes exists 
#     query_adata = format_query(query_adata,train_genes)
#     ref_adata = format_query(ref_adata,train_genes) 
    
#     reconstructed_query_adata = perform_inference(query_adata,ref_adata,model,args.enable_denovo)
    
#     # save the inferred data
#     reconstructed_query_adata.write(args.save_path)
   


# if __name__ == '__main__':
    
#     pass
