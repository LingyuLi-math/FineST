## 2024.01.24 copy from HIPT_image_feature.py
##            Adjust the code "for point in coordinates:", don't leave two spaces blank
## 2024.01.24 copy from HIPT_image_feature_NPC1new.py
##            Used to CRC
## 2024.10.05 copy from HIPT_image_feature_NPC1.py
##            Omit Image, only save Embedding

import os
import torch
from PIL import Image 
import numpy as np
import pandas as pd
Image.MAX_IMAGE_PIXELS = None
from torchvision import transforms
import random
import time


# Set seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(666)


# Define paths
path = '/home/lingyu/ssd/Python'
# path = '/mnt/lingyu/nfs_share2/Python/'
patient_image = 'HD_CRC_16um'
position_path = os.path.join(path, "FineST/FineST/Dataset/CRC16um/tissue_positions_square_016um.csv")
image_path = os.path.join(path, "FineST/FineST/Dataset/CRC16um/Visium_HD_Human_Colon_Cancer_tissue_image.btf")
patches_path = os.path.join(path, "FineST/HIPTtest/HD_CRC_16um_pth_32_16_image")
saved_pth_folder = os.path.join(path, "FineST/HIPTtest/HD_CRC_16um_pth_32_16")


# Load csv file with "no head"
tissue_position = pd.read_csv(position_path)    
##############################################
# different, need math with figure 
##############################################
x_positions = tissue_position["pxl_row_in_fullres"]
y_positions = tissue_position["pxl_col_in_fullres"]
coordinates = list(zip(x_positions, y_positions))

# https://github.com/mahmoodlab/HIPT/blob/master/HIPT_4K/hipt_4k.py
from HIPT.HIPT_4K import vision_transformer as vits

# Set device
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

# Load image
image = Image.open(image_path)
image_width, image_height = image.size
print("image_width, image_height: ", image_width, image_height)

# Create patches
patch_size = 32
os.makedirs(patches_path, exist_ok=True)

start_time = time.time()
for point in coordinates:
    x, y = point
    left = x - patch_size // 2
    upper = y - patch_size // 2
    right = x + patch_size // 2
    lower = y + patch_size // 2
    if left < 0 or upper < 0 or right > image_width or lower > image_height:
        continue
    patch = image.crop((left, upper, right, lower))
    patch_name = f"{patient_image}_{x}_{y}.png"
    print("patch_name: ", patch_name)
    patch.save(os.path.join(patches_path, patch_name))

end_time = time.time()
execution_time = end_time - start_time
print(f"The execution time for the loop is: {execution_time} seconds")

# https://github.com/mahmoodlab/HIPT/blob/a9b5bb8d159684fc4c2c497d68950ab915caeb7e/HIPT_4K/hipt_model_utils.py#L39
def get_vit256(pretrained_weights, arch='vit_small', device=torch.device('cuda:0')):
    r"""
    Builds ViT-256 Model.
    
    Args:
    - pretrained_weights (str): Path to ViT-256 Model Checkpoint.
    - arch (str): Which model architecture.
    - device (torch): Torch device to save model.
    
    Returns:
    - model256 (torch.nn): Initialized model.
    """
    
    checkpoint_key = 'teacher'
    device = torch.device("cpu")
    model256 = vits.__dict__[arch](patch_size=16, num_classes=0)
    for p in model256.parameters():
        p.requires_grad = False
    model256.eval()
    model256.to(device)

    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model256.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
        
    return model256

# Load model
weight_path = "https://github.com/mahmoodlab/HIPT/blob/master/HIPT_4K/Checkpoints/vit256_small_dino.pth"
model = get_vit256(pretrained_weights = weight_path)

# https://github.com/mahmoodlab/HIPT/blob/a9b5bb8d159684fc4c2c497d68950ab915caeb7e/HIPT_4K/hipt_model_utils.py#L111
def eval_transforms():
	mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
	eval_t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = mean, std = std)])
	return eval_t

# Process patches
os.makedirs(saved_pth_folder, exist_ok=True)
patches_list = os.listdir(patches_path)

for patch in patches_list:
    patch_base_name, extension = os.path.splitext(patch)
    patch_path = os.path.join(patches_path, patch)
    patch_image = Image.open(patch_path)
    p_image = eval_transforms()(patch_image).unsqueeze(dim=0)
    lay = model.get_intermediate_layers(p_image, 1)[0]
    subtensors = lay[:, :, :]
    subtensors_list = torch.split(subtensors, 1, dim=1)
    subtensors_list = subtensors_list[1:]
    saved_name = patch_base_name + '.pth'
    # print("saved_name: \n", saved_name)
    saved_path = os.path.join(saved_pth_folder, saved_name)
    torch.save(subtensors_list, saved_path)