# from cellContrast.model import *
# from cellContrast import utils
import numpy as np
import  logging
logging.getLogger().setLevel(logging.INFO)
from .utils import *
from .loadData import *


import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np


def calculate_correlation_infer(matrix_tensor_test_np, reconstructed_matrix_test_np, method="pearson"):
    correlation_coefficients = []

    for i in range(matrix_tensor_test_np.shape[0]):
        if method == "pearson":
            corr, _ = pearsonr(matrix_tensor_test_np[i], reconstructed_matrix_test_np[i])
        elif method == "spearman":
            corr, _ = spearmanr(matrix_tensor_test_np[i], reconstructed_matrix_test_np[i])
        else:
            raise ValueError("Invalid method, choose either 'pearson' or 'spearman'")
        correlation_coefficients.append(corr)

    mean_corr = np.mean(correlation_coefficients)
    return mean_corr

###########################################################################################
# Spot Correlation  
###########################################################################################
def calculate_correlation_spot(matrix_tensor_test, reconstructed_matrix_test, method="pearson"):
    matrix_tensor_test_np = matrix_tensor_test
    reconstructed_matrix_test_np = reconstructed_matrix_test

    correlation_coefficients = []

    for i in range(matrix_tensor_test_np.shape[0]):
        if method == "pearson":
            corr, _ = pearsonr(matrix_tensor_test_np[i], reconstructed_matrix_test_np[i])
        elif method == "spearman":
            corr, _ = spearmanr(matrix_tensor_test_np[i], reconstructed_matrix_test_np[i])
        else:
            raise ValueError("Invalid method, choose either 'pearson' or 'spearman'")
        correlation_coefficients.append(corr)

    return correlation_coefficients


def calculate_correlation_gene(matrix_tensor_test, reconstructed_matrix_test, method="pearson"):
    matrix_tensor_test_np = matrix_tensor_test
    reconstructed_matrix_test_np = reconstructed_matrix_test

    correlation_coefficients = []

    for i in range(matrix_tensor_test_np.shape[1]):
        if method == "pearson":
            corr, _ = pearsonr(matrix_tensor_test_np[:,i], reconstructed_matrix_test_np[:,i])
        elif method == "spearman":
            corr, _ = spearmanr(matrix_tensor_test_np[:,i], reconstructed_matrix_test_np[:,i])
        else:
            raise ValueError("Invalid method, choose either 'pearson' or 'spearman'")
        correlation_coefficients.append(corr)

    return correlation_coefficients


###########################################################################################
# Gene Correlation  
###########################################################################################
def calculate_correlation_infer_gene(matrix_tensor_test_np, reconstructed_matrix_test_np, method="pearson"):
    correlation_coefficients = []

    for i in range(matrix_tensor_test_np.shape[1]):
        if method == "pearson":
            ## 使用 NumPy 的 corrcoef 函数来计算相关系数
            corr = np.corrcoef(matrix_tensor_test_np[:,i], reconstructed_matrix_test_np[:,i])[0,1]
            ## 出现 nan
            # corr, _ = pearsonr(matrix_tensor_test_np[:,i], reconstructed_matrix_test_np[:,i])
        elif method == "spearman":
            corr, _ = spearmanr(matrix_tensor_test_np[:,i], reconstructed_matrix_test_np[:,i])
        else:
            raise ValueError("Invalid method, choose either 'pearson' or 'spearman'")
        correlation_coefficients.append(corr)
        
    mean_corr = np.nanmean(correlation_coefficients)
    return mean_corr

###########################################################################################
# cosine_similarity row (default)  
###########################################################################################
def calculate_cosine_similarity_row(rep_query_adata, rep_ref_adata_image_reshape):
    if isinstance(rep_query_adata, torch.Tensor):
        rep_query_adata = rep_query_adata.numpy()

    if isinstance(rep_ref_adata_image_reshape, torch.Tensor):
        rep_ref_adata_image_reshape = rep_ref_adata_image_reshape.numpy()

    cosine_sim = cosine_similarity(rep_query_adata, rep_ref_adata_image_reshape)
    
    return cosine_sim


def calculate_cosine_similarity_col(rep_query_adata, rep_ref_adata_image_reshape):
    if isinstance(rep_query_adata, torch.Tensor):
        rep_query_adata = rep_query_adata.numpy()

    if isinstance(rep_ref_adata_image_reshape, torch.Tensor):
        rep_ref_adata_image_reshape = rep_ref_adata_image_reshape.numpy()

    # 转置矩阵，使得列变为行
    rep_query_adata_T = rep_query_adata.T
    rep_ref_adata_image_reshape_T = rep_ref_adata_image_reshape.T

    # 计算余弦相似性
    cosine_sim = cosine_similarity(rep_query_adata_T, rep_ref_adata_image_reshape_T)
    
    return cosine_sim


def compute_corr(expression_gt, matched_spot_expression_pred, top_k=50, qc_idx=None):
    ## 计算基因表达数据（expression_gt）和预测数据（matched_spot_expression_pred）之间，排名前 50 的相关性
    ## cells are in columns, genes are in rows
    if qc_idx is not None:
        expression_gt = expression_gt[:, qc_idx]
        matched_spot_expression_pred = matched_spot_expression_pred[:, qc_idx]

    mean = np.mean(expression_gt, axis=1)
    top_genes_idx = np.argpartition(mean, -top_k)[-top_k:]

    # 使用列表推导式计算相关性
    corr = [np.corrcoef(expression_gt[i, :], matched_spot_expression_pred[i, :])[0, 1] for i in top_genes_idx]

    return np.mean(corr)


###########################################################################################
# Three correlation for spot and gene: output/  ## which are in notebook
###########################################################################################
def calculate_statistics(matrix1, matrix2, label):
    print(matrix1.shape)
    print(matrix2.shape)

    ## pearson
    mean_pearson_corr = calculate_correlation_infer(matrix1, matrix2, method="pearson")
    print(f"Mean Pearson correlation coefficient--{label}: {mean_pearson_corr:.4f}")
    logger.info(f'Mean Pearson correlation coefficient--{label}: {mean_pearson_corr}')

    ## spearman
    mean_spearman_corr = calculate_correlation_infer(matrix1, matrix2, method="spearman")
    print(f"Mean Spearman correlation coefficient--{label}: {mean_spearman_corr:.4f}")
    logger.info(f'Mean Spearman correlation coefficient--{label}: {mean_spearman_corr}')

    ## cosine_similarity_row
    cosine_sim = calculate_cosine_similarity_row(matrix1, matrix2)
    cosine_sim_per_sample = np.diag(cosine_sim)
    average_cosine_similarity = np.mean(cosine_sim_per_sample)

    print(f"Average cosine similarity--{label}: {average_cosine_similarity:.4f}")
    logger.info(f'Average cosine similarity--{label}: {average_cosine_similarity}')


def calculate_statistics_gene(matrix1, matrix2, label):
    print(matrix1.shape)
    print(matrix2.shape)

    mean_pearson_corr = calculate_correlation_infer_gene(matrix1, matrix2, method="pearson")
    print(f"Mean Pearson correlation coefficient--{label}: {mean_pearson_corr:.4f}")
    logger.info(f'Mean Pearson correlation coefficient--{label}: {mean_pearson_corr}')
    
    mean_spearman_corr = calculate_correlation_infer_gene(matrix1, matrix2, method="spearman")
    print(f"Mean Spearman correlation coefficient--{label}: {mean_spearman_corr:.4f}")
    logger.info(f'Mean Spearman correlation coefficient--{label}: {mean_spearman_corr}')

    cosine_sim = calculate_cosine_similarity_col(matrix1, matrix2)
    cosine_sim_per_sample = np.diag(cosine_sim)
    average_cosine_similarity = np.mean(cosine_sim_per_sample)
    
    print(f"Average cosine similarity--{label}: {average_cosine_similarity:.4f}")
    logger.info(f'Average cosine similarity--{label}: {average_cosine_similarity}')


    