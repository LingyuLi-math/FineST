import scanpy as sc
# from cellContrast.model import *
# from cellContrast import utils
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
