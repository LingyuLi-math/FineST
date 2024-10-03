"""
Utils of permutation calculation
"""
import numpy as np
import random
import pandas as pd
import torch
import time
import logging
import os


## set the random seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


## set the logging
def setup_logger(model_save_folder):
        
    level =logging.INFO

    log_name = 'model.log'
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(model_save_folder + log_name)
    logger.setLevel(level)
    
    fileHandler = logging.FileHandler(os.path.join(model_save_folder, log_name), mode = 'a')
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)

    return logger


## set the device
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


## define function
def reshape_latent_image(inputdata):    ## [adata.shape[0]*256, 384]  -->  [adata.shape[0], 384]
    inputdata_reshaped = inputdata.view(int(inputdata.shape[0]/16), 16, inputdata.shape[1])  # [adata.shape[0], 256, 384]
    average_inputdata_reshaped = torch.sum(inputdata_reshaped, dim=1) / inputdata_reshaped.size(1)
    return inputdata_reshaped, average_inputdata_reshaped
