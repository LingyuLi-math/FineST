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
from .loadData import *
from scipy.spatial import cKDTree



## Find the nearest point in adata_know for each point in adata_spot
def find_nearest_point(adata_spot, adata_know):
    nearest_points = []
    for point in adata_spot:
        distances = np.linalg.norm(adata_know - point, axis=1)
        nearest_index = np.argmin(distances)
        nearest_points.append(adata_know[nearest_index])
    return np.array(nearest_points)

##################################################################################
# using 7 neighborhood： cKDTree is more faster
# Find k nearest neighbors for each point in nearest_points within adata_know
##################################################################################

## Function 1: Using cKDTree
def find_nearest_neighbors(nearest_points, adata_know, k=6):
    nbs = []
    nbs_indices = []
    tree = cKDTree(adata_know)
    for point in nearest_points:
        dist, indices = tree.query(point, k+1)
        nbs.append(adata_know[indices])
        nbs_indices.append(indices)
    return np.array(nbs), np.array(nbs_indices)

# ## Function 2: Using NearestNeighbors
# from sklearn.neighbors import NearestNeighbors
# def find_nearest_neighbors(nearest_points, adata_know, k=6):
#     nbs = []
#     nbs_indices = []
#     nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', metric='euclidean').fit(adata_know)
#     for point in nearest_points:
#         distances, indices = nbrs.kneighbors([point])
#         nbs.append(adata_know[indices][0])
#         nbs_indices.append(indices[0])
#     return np.array(nbs), np.array(nbs_indices)

## Calculate Euclidean distances between each point in adata_spot and its nearest neighbors
def calculate_euclidean_distances(adata_spot, nbs):
    distances = []
    for point, neighbors in zip(adata_spot, nbs):
        dist = np.linalg.norm(neighbors - point, axis=1)
        distances.append(dist)
    return np.array(distances)




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
