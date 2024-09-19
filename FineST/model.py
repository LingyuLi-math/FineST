import torch
from torch import nn
import torch.nn.functional as F
from .utils import *
import os
import json
import numpy as np
import  logging
logging.getLogger().setLevel(logging.INFO)


# ## set the device
# if torch.cuda.is_available():
#     dev = "cuda:0"
# else:
#     dev = "cpu"
# device = torch.device(dev)


# ## Set the random seed for PyTorch and NumPy
# torch.manual_seed(666)


#######################################################################
## 2024.9.16 LLY add some function for Train and Test
#######################################################################

class DatasetCreat(torch.utils.data.Dataset):
    def __init__(self, image_paths, spatial_pos_path, reduced_mtx_path):
        self.spatial_pos_csv = pd.read_csv(spatial_pos_path, sep=",", header=None)
        # 加载表达矩阵（转置为 cell x features）
        self.reduced_matrix = np.load(reduced_mtx_path).T    # 若有行名和列名，使用 allow_pickle=True
        
        # 加载 .pth 文件
        self.images = []
        for image_path in image_paths:
            if image_path.endswith('.pth'):
                image_tensor = torch.load(image_path)
                self.images.extend(image_tensor)
        self.image_data = torch.stack(self.images)
        self.image_tensor = self.image_data.view(self.image_data.size(0), -1)  
                
        print("Finished loading all files")

    def __getitem__(self, idx):
        item = {}
        v1 = self.spatial_pos_csv.loc[idx, 0]   
        v2 = self.spatial_pos_csv.loc[idx, 1]  
        v3 = self.spatial_pos_csv.loc[idx, 2]   
        v4 = self.spatial_pos_csv.loc[idx, 3]  
    
        # Stack the tensors in the list along a new dimension  
        item['image'] = self.image_tensor[idx * 16 : (idx + 1) * 16]    
        item['reduced_expression'] = torch.tensor(self.reduced_matrix[idx, :]).float()  # 提取对应的表达矩阵数据
        item['spatial_coords'] = [v1, v2]  
        item['array_row'] = v3   
        item['array_col'] = v4  
    
        return item

    def __len__(self):
        return len(self.reduced_matrix)
    



## 2023.10.26 need add CAE loss  
# class ContrastiveLoss(nn.Module):
#     def __init__(self, temperature=0.1):
#         super(ContrastiveLoss, self).__init__()
#         self.temperature = temperature

#     def forward(self, embeddings, labels):

#         batch_size = embeddings.shape[0]
        
#         embeddings = F.normalize(embeddings, p=2, dim=1)
#         sim_matrix = torch.exp(torch.matmul(embeddings, embeddings.T) / self.temperature)
       
#         pos_mask = (labels == 1).type(torch.bool)

#         neg_mask = ~pos_mask
#         neg_mask.fill_diagonal_(False)  # remove diagonal items

#         pos_similarities = (sim_matrix * pos_mask).sum(dim=1)
#         neg_similarities = (sim_matrix * neg_mask).sum(dim=1)
        
#         loss = -torch.mean(torch.log(pos_similarities / neg_similarities))

#         return loss


###############################
# CoSimLoss Loss： used
###############################
def CoSimLoss(Y_hat: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    # Initialize cosine similarity objects for column-wise and row-wise calculations
    cos_by_col = nn.CosineSimilarity(dim=1)
    cos_by_row = nn.CosineSimilarity(dim=0)

    # Compute column-wise cosine similarity loss between Y_hat and Y
    loss1 = (1 - cos_by_col(Y_hat, Y)).mean()

    # Compute row-wise cosine similarity loss between Y_hat and Y
    loss2 = (1 - cos_by_row(Y_hat, Y)).mean()

    # Combine the two losses to get the final loss (imp_loss)
    imp_loss = loss1 + loss2

    return imp_loss


###############################
# PearsonCorrelationLoss： v1
###############################
# def PearsonCorrelationLoss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#     def compute_pearson_corr(x, y, dim):
#         x_mean = torch.mean(x, dim=dim, keepdim=True)
#         y_mean = torch.mean(y, dim=dim, keepdim=True)
#         x_std = torch.std(x, dim=dim, keepdim=True)
#         y_std = torch.std(y, dim=dim, keepdim=True)
#         cov = torch.mean((x - x_mean) * (y - y_mean), dim=dim, keepdim=True)
#         pearson_corr = cov / (x_std * y_std)
#         loss = (1 - pearson_corr) / 2
#         return torch.mean(loss)

#     row_loss = compute_pearson_corr(x, y, dim=1)
#     col_loss = compute_pearson_corr(x, y, dim=0)
    
#     combined_loss = row_loss + col_loss
#     return combined_loss


###############################
# PearsonCorrelationLoss： v2
###############################
# def PearsonCorrelationLoss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#     def compute_pearson_corr(x, y):
#         x_mean = torch.mean(x, dim=0, keepdim=True)
#         y_mean = torch.mean(y, dim=0, keepdim=True)
#         x_std = torch.std(x, dim=0, keepdim=True)
#         y_std = torch.std(y, dim=0, keepdim=True)
#         cov = torch.mean((x - x_mean) * (y - y_mean), dim=0, keepdim=True)
#         pearson_corr = cov / (x_std * y_std)
#         loss = (1 - pearson_corr) / 2
#         return torch.mean(loss)
        
#     col_loss = compute_pearson_corr(x, y)
#     return col_loss


###############################
# ContrastiveLoss Loss： used
###############################
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings_iamge,
        embeddings_matrix,
        labels,
        input_image_exp,
        reconstruction_iamge,
        reconstructed_matrix_all,
        reconstruction_iamge_reshapef2,
        input_matrix_exp,
        # w1, w2, w3, w4
        w1=1, w2=1, w3=1, w4=1
    ): 

        # 对嵌入向量进行 L2 范数归一化
        embeddings_iamge = F.normalize(embeddings_iamge, p=2, dim=1)
        embeddings_matrix = F.normalize(embeddings_matrix, p=2, dim=1)
        
        # 计算嵌入向量之间的相似性矩阵（使用指数除以 temperature 参数）
        sim_matrix = torch.exp(torch.matmul(embeddings_iamge, embeddings_matrix.T) / self.temperature)
        
        # 创建正向相似性掩码（对应标签为 1 的位置）
        pos_mask = (labels == 1).type(torch.bool)
        # 创建负向相似性掩码（对应标签不为 1 的位置）
        neg_mask = ~pos_mask
        # neg_mask.fill_diagonal_(False)  # 移除对角线元素


        pos_similarities = (sim_matrix * pos_mask).sum(dim=1)
        neg_similarities = (sim_matrix * neg_mask).sum(dim=1)
        
        #################################################################################################
        ## 2023.12.20 adjust loss 

        # ## 计算最终损失值，取负对数似然
        # loss = (-w1*torch.mean(torch.log(pos_similarities / neg_similarities))    # contrast loss
        #         + w2*PearsonCorrelationLoss(reconstruction_iamge, input_image_exp)          # image loss
        #         + w3*PearsonCorrelationLoss(reconstruction_iamge_reshapef2, reconstructed_matrix_all)   # cross loss
        #         + w4*PearsonCorrelationLoss(input_matrix_exp, reconstructed_matrix_all))    # matirx loss
        #################################################################################################
        ## 2023.12.15 adjust loss 

        ## 计算最终损失值，取负对数似然
        loss = (-w1*torch.mean(torch.log(pos_similarities / neg_similarities))    # contrast loss
                + w2*CoSimLoss(reconstruction_iamge, input_image_exp)          # image loss
                + w3*CoSimLoss(reconstruction_iamge_reshapef2, reconstructed_matrix_all)   # cross loss
                + w4*CoSimLoss(input_matrix_exp, reconstructed_matrix_all))    # matirx loss
        #################################################################################################
        
        # 计算最终损失值，取负对数似然
        # loss = (-w1*torch.mean(torch.log(pos_similarities / neg_similarities))    # contrast loss
        #         + w2*nn.MSELoss()(reconstruction_iamge, input_image_exp)          # image loss
        #         + w3*nn.MSELoss()(reconstruction_iamge_reshapef2, reconstructed_matrix_all)   # cross loss
        #         + w4*nn.MSELoss()(input_matrix_exp, reconstructed_matrix_all))    # matirx loss

        return loss
    

###################
# ProjectionHead  
###################
class ProjectionHead(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)   
        # self.gelu = nn.GELU()  
        self.relu = nn.ReLU(inplace=True)  
        self.fc2 = nn.Linear(hidden_size, output_size)   

    def forward(self, x):
        x = self.fc1(x)  
        # x = self.gelu(x)   # BLEEP
        x = self.relu(x)  
        x = self.fc2(x)   
        return x
      
    

class ELU(nn.Module):

    def __init__(self, alpha, beta):
        super().__init__()
        self.activation = nn.ELU(alpha=alpha, inplace=True)
        self.beta = beta

    def forward(self, x):
        return self.activation(x) + self.beta

##############################
# Spatial data embedding  
##############################
class Autoencoder_matrix(nn.Module):    # 2 layers
    def __init__(self, input_dim, latent_dim, n_encoder_hidden_matrix, negative_slope=0.01):
        super(Autoencoder_matrix, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, n_encoder_hidden_matrix),
            nn.LeakyReLU(negative_slope),    # GELU()  BLEEP
            nn.Linear(n_encoder_hidden_matrix, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, n_encoder_hidden_matrix),
            nn.LeakyReLU(negative_slope),
            nn.Linear(n_encoder_hidden_matrix, input_dim),
            # nn.Sigmoid(),
            # nn.Softplus(),     # xfuse   positive
            ELU(alpha=0.01, beta=0.01),    # iSTAR     positive
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


###############################
# Patch image embedding  
###############################
class Autoencoder_image(nn.Module):
    def __init__(self, input_dim, latent_dim, n_encoder_hidden_image, negative_slope=0.01):
        super(Autoencoder_image, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, n_encoder_hidden_image),
            nn.LeakyReLU(negative_slope),
            nn.Linear(n_encoder_hidden_image, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, n_encoder_hidden_image),
            nn.LeakyReLU(negative_slope),
            nn.Linear(n_encoder_hidden_image, input_dim),
            # nn.Softplus(),     # xfuse   positivate
            ELU(alpha=0.01, beta=0.01),    # iSTAR     positivate
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

########################
# CellContrastModel    
########################
class CellContrastModel(nn.Module):
    def __init__(
        self, 
        n_input_matrix=578, 
        n_input_image=384, 
        n_encoder_hidden_matrix=512,
        n_encoder_hidden_image=256,
        n_encoder_hidden=512,  # not use
        n_encoder_latent=128, 
        n_projection_hidden=256, 
        n_projection_output=128, 
        n_encoder_layers=2, 
        dropout_rate=0, 
        negative_slope=0.01
    ):
        super(CellContrastModel,self).__init__()     
        self.Autoencoder_matrix = Autoencoder_matrix(n_input_matrix, n_encoder_latent, n_encoder_hidden_matrix) 
        self.matrix_encoder = self.Autoencoder_matrix.encoder
        self.matrix_decoder = self.Autoencoder_matrix.decoder
        self.Autoencoder_image = Autoencoder_image(n_input_image, n_encoder_latent, n_encoder_hidden_image)  
        self.image_encoder = self.Autoencoder_image.encoder
        self.image_decoder = self.Autoencoder_image.decoder
        self.image_projection = ProjectionHead(n_encoder_latent, n_projection_hidden, n_projection_output)
        self.matrix_projection = ProjectionHead(n_encoder_latent, n_projection_hidden, n_projection_output)
            
    def forward(self, x, y):              
        representation_matrix = self.matrix_encoder(x) 
        reconstruction_matrix = self.matrix_decoder(representation_matrix) 
        projection_matrix = self.matrix_projection(representation_matrix)       
        representation_image = self.image_encoder(y) 
        reconstruction_image = self.image_decoder(representation_image) 
        projection_image = self.image_projection(representation_image)      
        return representation_matrix, reconstruction_matrix, projection_matrix, representation_image, reconstruction_image, projection_image



def load_model(dir_name, parameter_file_path, params, gene_hv):    
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



# if __name__ == '__main__':
    
#     model = CellContrastModel(n_input=351, n_encoder_hidden=1024,n_encoder_latent=512,n_encoder_layers=2,n_projection_hidden=256,n_projection_output=128)
#     print(model)
    
#     pass

