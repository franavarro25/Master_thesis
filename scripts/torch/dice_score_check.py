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
from data_utils import load_data, data_loader
from utils import run_epoch
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

# code to get y_source and target
dataset, transformations = load_data('/home/franavarro25/TUM/Master_thesis/dataset/OASIS/train/**/') 
train_loader, val_loader = data_loader(dataset=dataset, batch_size=1) 
data = next(iter(train_loader))
in_shape = data['image'][0].shape[-3:]

device = 'cuda'
model_dir = '/home/franavarro25/TUM/Master_thesis/voxelmorph_monai/models/presentation/(bw0.5)_cobi_patches_local/lambda_1/0500.pt'
model = vxm.networks.VxmDense.load(model_dir, device).cuda()

transform_model = vxm.layers.SpatialTransformer(in_shape, mode='nearest').to(device)


for i, data in enumerate(train_loader):
    with torch.no_grad():
        moved_mask, fixed_mask, target, y_source, flow = run_epoch(
                    data=data, model=model, transform_model=transform_model, device=device, task='val')
        
        dice_labels = monai.metrics.meandice.compute_dice(y_pred=moved_mask, y=fixed_mask)
        dice_step = torch.mean(dice_labels)
        print(dice_step, dice_labels) 

        '''loaded_source = sitk.GetImageFromArray(source)
        sitk.WriteImage(loaded_source, '/home/franavarro25/Desktop/loaded_images/'+str(i)+'_source.nii.gz')
        loaded_target = sitk.GetImageFromArray(target)
        sitk.WriteImage(loaded_target, '/home/franavarro25/Desktop/loaded_images/'+str(i)+'_target.nii.gz')
        loaded_source_mask = sitk.GetImageFromArray(source_mask)
        sitk.WriteImage(loaded_source_mask, '/home/franavarro25/Desktop/loaded_images/'+str(i)+'_source_mask.nii.gz')
        loaded_target_mask = sitk.GetImageFromArray(target_mask)
        sitk.WriteImage(loaded_target_mask, '/home/franavarro25/Desktop/loaded_images/'+str(i)+'_target_mask.nii.gz')'''

        in_feat_shape_x = y_source.shape[-3:]
        in_feat_shape_y = target.shape[-3:]
        x = y_source 
        y = target
        mid_slices_moved = [np.take(x.detach().cpu().squeeze(0), in_feat_shape_x[d]//2, axis=d+1) for d in range(3)]
        mid_slices_moved[1] = np.rot90(mid_slices_moved[1], 1)
        mid_slices_moved[2] = np.rot90(mid_slices_moved[2], -1)
        print(mid_slices_moved[0].shape, mid_slices_moved[1].shape, mid_slices_moved[2].shape)

        mid_slices_fixed = [np.take(y.detach().cpu().squeeze(0), in_feat_shape_y[d]//2, axis=d+1) for d in range(3)]
        mid_slices_fixed[1] = np.rot90(mid_slices_fixed[1], 1)
        mid_slices_fixed[2] = np.rot90(mid_slices_fixed[2], -1)
        print(mid_slices_fixed[0].shape, mid_slices_fixed[1].shape, mid_slices_fixed[2].shape)

        cmap_color = 'gray'
        # plot the feature map
        plt.figure(figsize=(8,8))
        plt.imshow(mid_slices_moved[0][0], cmap=cmap_color)
        plt.axis('off')
        plt.title('moved')

        plt.figure(figsize=(8,8))
        plt.imshow(mid_slices_fixed[0][0], cmap=cmap_color)
        plt.axis('off')
        plt.title('fixed')

        #plt.show()

        # plot the feature map
        plt.figure(figsize=(8,8))
        plt.imshow(mid_slices_moved[1][:,0,:], cmap=cmap_color)
        plt.axis('off')
        plt.title('moved')

        plt.figure(figsize=(8,8))
        plt.imshow(mid_slices_fixed[1][:,0,:], cmap=cmap_color)
        plt.axis('off')
        plt.title('fixed')

        #plt.show()

        # plot the feature map
        plt.figure(figsize=(8,8))
        plt.imshow(mid_slices_moved[2][:,0,:], cmap=cmap_color)
        plt.axis('off')
        plt.title('moved')

        plt.figure(figsize=(8,8))
        plt.imshow(mid_slices_fixed[2][:,0,:], cmap=cmap_color)
        plt.axis('off')
        plt.title('fixed')

        plt.show()

       