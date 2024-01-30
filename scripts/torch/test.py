#!/usr/bin/env python

"""
Example script to register two volumes with VoxelMorph models.

Please make sure to use trained models appropriately. Let's say we have a model trained to register 
a scan (moving) to an atlas (fixed). To register a scan to the atlas and save the warp field, run:

    register.py --moving moving.nii.gz --fixed fixed.nii.gz --model model.pt 
        --moved moved.nii.gz --warp warp.nii.gz

The source and target input images are expected to be affinely registered.

If you use this code, please cite the following, and read function docs for further info/citations
    VoxelMorph: A Learning Framework for Deformable Medical Image Registration 
    G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. 
    IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019. 

    or

    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu. 
    MedIA: Medical Image Analysis. (57). pp 226-236, 2019 

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in 
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or 
implied. See the License for the specific language governing permissions and limitations under 
the License.
"""

import os
import argparse

# third party
import numpy as np
import nibabel as nib
import torch
import monai
import wandb

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import sys 
sys.path.append('/home/franavarro25/TUM/Master_thesis/voxelmorph_monai')
import voxelmorph as vxm   # nopep8

# import my utils
from data_utils import load_data, data_loader
from vxm_network import vxm_model
from losses import loss_functions
from utils import run_epoch
from metrics import SDlogDetJac

# parse commandline args
parser = argparse.ArgumentParser()

# Using wandb
parser.add_argument('--wandb', default=False, type=bool, help='save training values with weights and biases')
parser.add_argument('--name', default='evaluation_2', help='name of the wandb run')

# data an model
parser.add_argument('--data-dir', default='/home/franavarro25/TUM/Master_thesis/dataset/OASIS/test/**/', help='line-seperated list of training files')
parser.add_argument('--model',  default='/home/franavarro25/TUM/Master_thesis/voxelmorph_monai/models/mse/2/0100.pt', 
                    help='pytorch model for nonlinear registration')
                    # best lambdas--> mind/lambda_0.1/0100.pt; mi/lambda_3.0/0100.pt; ngf/lambda_0.001/0100.pt; mse/2/0100.pt; ncc/2/0100.pt
parser.add_argument('-g', '--gpu', default=0, help='GPU number(s) - if not supplied, CPU is used')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
# loss hyperparameters
parser.add_argument('--image-loss', required=True,
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', required=True,
                    help='weight of deformation loss (default: 0.01)')
                    # best lambdas--> mind=0.1; mi=3; ngf=0.001
args = parser.parse_args()
print('args', args)

# Initialize wandb
if args.wandb:
    run = wandb.init(project='voxelmorph',group=args.name, name=args.image_loss+'_lambda='+str(args.weight),
    config={'type of loss':args.image_loss, 'model':args.model, 'lambda':args.weight})

# device handling
if args.gpu and (args.gpu != '-1'):
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print('device', device)
else:
    device = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# load moving and fixed images
# load and prepare test data
# dict with keys:image,label;values:file_names 
dataset, transformations = load_data(args.data_dir) 

# dataloader--> dict with keys: image,labels,image_metadata,label_metadata; values: real 3d volumes 
test_loader,_= data_loader(dataset=dataset, batch_size=1) 

inshape = next(iter(test_loader))['image'].shape[-3:]


# load and set up model
model = vxm.networks.VxmDense.load(args.model, device)
model.to(device)
model.eval()
transform_model = vxm.layers.SpatialTransformer(inshape, mode='nearest').to(device)


# Metrics
compute_jackdet = SDlogDetJac()
dice_score = []
hausdorff_distance = []
SDlog_Jac = []
non_pos_jacdet = []

#evaluation
for data in test_loader:
    moved_mask, fixed_mask, flow = run_epoch(
                    data=data, model=model, transform_model=transform_model,device=device, task='val')

    sdlog_jacdet_step, non_pos_jacdet_step = compute_jackdet(flow.detach().cpu())
    dice_step = torch.mean(monai.metrics.meandice.compute_dice(y_pred=moved_mask, y=fixed_mask)) 
    hausdorff_distance_step = torch.mean(monai.metrics.hausdorff_distance.compute_hausdorff_distance(moved_mask, fixed_mask))
    print('values', sdlog_jacdet_step, non_pos_jacdet_step, dice_step, hausdorff_distance_step)
    if  args.wandb:
        wandb.log({'dice_score(val)_step':dice_step.item(),
                    'hausdorff_distance_step ':hausdorff_distance_step.item(),
                    'SDlog_Jac_step':sdlog_jacdet_step.item(),
                    'non_pos_jacdet_step':non_pos_jacdet_step.item()})#*100



    SDlog_Jac.append(sdlog_jacdet_step)
    non_pos_jacdet.append(non_pos_jacdet_step)
    dice_score.append(dice_step)
    hausdorff_distance.append(hausdorff_distance_step)

# mean values 
dice_score = sum(dice_score)/len(dice_score)
hausdorff_distance = sum(hausdorff_distance)/len(hausdorff_distance)
SDlog_Jac = sum(SDlog_Jac)/len(SDlog_Jac)
non_pos_jacdet = sum(non_pos_jacdet)/len(non_pos_jacdet)


if  args.wandb:
        wandb.log({'dice_score(val)':dice_score.item(),
                    'hausdorff_distance ':hausdorff_distance.item(),
                    'SDlog_Jac':SDlog_Jac.item(),
                    'non_pos_jacdet':non_pos_jacdet.item()})

print('Dice_score', dice_score)
print('Hsdrf_dist', hausdorff_distance)
print('SDlog_Jac', SDlog_Jac)
print('non_pos_jacdet', non_pos_jacdet)
