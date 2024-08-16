import torch
from torch import nn
import torch.nn.functional as F
from .utils import *

## Set the random seed for PyTorch and NumPy
torch.manual_seed(0)


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
    

# class ProjectionHead(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(ProjectionHead, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x


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
      
    
# class Encoder(nn.Module):
#     def __init__(self, n_input, n_latent, n_layers=2, n_hidden=1024, dropout_rate=0,negative_slope=0.01):
        
#         super().__init__()

#         self.n_input = n_input
#         self.n_latent = n_latent
#         self.n_layers = n_layers
#         self.n_hidden = n_hidden
#         self.dropout_rate = dropout_rate
#         self.negative_slope = negative_slope

#         # Define the hidden layers
#         self.hidden_layers = nn.ModuleList()
#         for i in range(self.n_layers - 1):
#             if(i==0):
#                 self.hidden_layers.append(nn.Linear(self.n_input, self.n_hidden))
#             else:
#                 self.hidden_layers.append(nn.Linear(self.n_hidden, self.n_hidden))
#             self.hidden_layers.append(nn.BatchNorm1d(self.n_hidden))
#             self.hidden_layers.append(nn.LeakyReLU(self.negative_slope))
#             self.hidden_layers.append(nn.Dropout(self.dropout_rate))

#         # Define the output layer
#         self.output_layer = nn.Linear(self.n_hidden, self.n_latent)
        
#         # Define additional layers after the output layer
#         self.bn_layer = nn.BatchNorm1d(self.n_latent)
#         self.dropout_layer = nn.Dropout(self.dropout_rate)
#         self.activation_layer = nn.LeakyReLU(self.negative_slope)

#     def forward(self, x):
        
#         # Pass input through the hidden layers
#         for hidden_layer in self.hidden_layers:
#             x = hidden_layer(x)

        
#         # Compute the representations
#         embeddings = self.output_layer(x)

#         embeddings = self.bn_layer(embeddings)
#         embeddings = self.activation_layer(embeddings)
#         embeddings = self.dropout_layer(embeddings)
        

#         return embeddings


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


# class CellContrastModel(nn.Module):


#     def __init__(self,n_input, n_encoder_hidden=1024,n_encoder_latent=512,n_encoder_layers=2,\
#                  n_projection_hidden=256,n_projection_output=128,dropout_rate=0,negative_slope=0.01):

#         super(CellContrastModel,self).__init__()
     
#         self.encoder = Encoder(n_input, n_encoder_latent, n_encoder_layers, n_encoder_hidden, dropout_rate,negative_slope)
        
#         self.projection = ProjectionHead(n_encoder_latent, n_projection_hidden, n_projection_output)

        

#     def forward(self,x):
        
#         representation = self.encoder(x)
#         projection = self.projection(representation)
        
#         return representation, projection
    

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



# if __name__ == '__main__':
    
#     model = CellContrastModel(n_input=351, n_encoder_hidden=1024,n_encoder_latent=512,n_encoder_layers=2,n_projection_hidden=256,n_projection_output=128)
#     print(model)
    
#     pass

