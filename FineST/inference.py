import  logging
logging.getLogger().setLevel(logging.INFO)
from .utils import *
from .loadData import *



# # set the device
# if torch.cuda.is_available():
#     dev = "cuda:0"
# else:
#     dev = "cpu"
# device = torch.device(dev)



def perform_inference(model, test_loader):
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
    image_profile_reshape = image_profile.view(-1, image_profile.shape[2])     # [adata.shape[0], 256, 384] --> [adata.shape[0]*256, 384]
    input_image_exp = image_profile_reshape.clone().detach().to(device)     # SDU

    ## model
    (representation_matrix, 
     reconstructed_matrix, 
     projection_matrix,
     representation_image, 
     reconstruction_iamge, 
     projection_image) = model(matrix_profile, input_image_exp)     
    
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



def perform_inference_image_between_spot(model, test_loader):
    print("device",device)    

    #####################################################################################
    # for whole dataset
    #####################################################################################        
    print("***** Begin perform_inference: ******")
    
    input_image_all, input_coord_all = extract_test_data_image_between_spot(test_loader)   
            
    ## input image
    image_profile = input_image_all.to(device)
    ## reshape image
    image_profile_reshape = image_profile.view(-1, image_profile.shape[2])     # [adata.shape[0], 256, 384] --> [adata.shape[0]*256, 384]
    input_image_exp = image_profile_reshape.clone().detach().to(device)     # SDU
    ## useful model
    representation_image = model.image_encoder(input_image_exp) 
    ## cross decoder
    reconstructed_matrix_reshaped = model.matrix_decoder(representation_image)  
    _, reconstruction_iamge_reshapef2 = reshape_latent_image(reconstructed_matrix_reshaped)
    ## reshape
    _, representation_image_reshape = reshape_latent_image(representation_image)
    
    #####################################################################################  
    # convert
    #####################################################################################  
    ## matrix
    representation_image_reshape = representation_image_reshape.cpu().detach().numpy() 
    reconstruction_iamge_reshapef2 = reconstruction_iamge_reshapef2.cpu().detach().numpy() 
    
    return (reconstruction_iamge_reshapef2, 
            reconstructed_matrix_reshaped,
            representation_image_reshape,
            input_image_exp,
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
