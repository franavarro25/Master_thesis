import os
# import voxelmorph with pytorch backend from source
os.environ['NEURITE_BACKEND'] = 'pytorch'  ###############333
os.environ['VXM_BACKEND'] = 'pytorch'      ############
import sys 
sys.path.append('/home/franavarro25/TUM/Master_thesis/voxelmorph_monai')
import voxelmorph as vxm
import monai
import torch
from vxm_network import vxm_model
import argparse
from data_utils import load_data_multimodal, data_loader_multimodal, BrainOneHotd
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from pathlib import Path

# get the folders that contain seg
T1_names, T2_names, uids = [],[],[]
t1_dir = '/home/franavarro25/TUM/Master_thesis/dataset/ADNI_remote/ADNI_T1_MNIspace/'
data_dir = '/home/franavarro25/TUM/Master_thesis/dataset/ADNI_remote/validation/'
path = Path(data_dir)
for file in path.rglob('aseg_moved.nii.gz'):
    if file.is_file() :
        T1_names.append(t1_dir + str(file).split('/')[-2].split('_')[0] + '/aseg.nii.gz')
        #T2_names.append(t1_dir + str(file).split('/')[-2].split('_')[1] + '/aseg.nii.gz')
        T2_names.append(str(file))
        uids.append([str(file).split('/')[-2].split('_')[0], str(file).split('/')[-2].split('_')[1]])
print(uids)
print(len(T1_names), len(T2_names))

# one hot class
brainOneHotd = BrainOneHotd(keys='seg')

total_dsc = []
for i, (t1_uid_path, t2_uid_path) in enumerate(zip(T1_names,T2_names)):
    print(i, "\n T1 values:", t1_uid_path, "\n T2 values:", t2_uid_path)
    # code to get y_source and target
    image_1 = nib.load(t1_uid_path).get_fdata()
    image_2 = nib.load(t2_uid_path).get_fdata() #878146
    #print(np.unique(image_1), np.unique(image_2)) 
    print(image_1.shape)

    source_mask = {'seg': image_1}
    target_mask = {'seg': image_2}

    # binarize mask (20x36x160x192x224)
    source_mask = brainOneHotd(source_mask)['seg'].unsqueeze(0)
    target_mask = brainOneHotd(target_mask)['seg'].unsqueeze(0)
    #print(torch.unique(source_mask), source_mask.shape)
    dice_score_labels = monai.metrics.meandice.compute_dice(y_pred=source_mask, y=target_mask, include_background=False)
    total_dsc.append(dice_score_labels)
    print(dice_score_labels, dice_score_labels.mean())
total_dsc = torch.cat(total_dsc)
print(total_dsc.mean())
        