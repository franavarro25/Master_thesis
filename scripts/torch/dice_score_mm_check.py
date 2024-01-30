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

# code to get y_source and target
dataset = load_data_multimodal(data_dir='/home/franavarro25/TUM/Master_thesis/dataset/ADNI_remote/ADNI_T1_MNIspace/', 
                               csv_file = '/home/franavarro25/TUM/Master_thesis/ANTs_cpp/non_valid_uids.csv') 
training_loader, val_loader = data_loader_multimodal(dataset = dataset, 
                                                     batch_size=1)

brainOneHotd = BrainOneHotd(keys='seg')

data = next(iter(training_loader))
device = 'cpu'
source = data['image2'][1].unsqueeze(0).float().to(device)
target = data['image1'][0].unsqueeze(0).float().to(device)
source_mask = data['seg'][1].unsqueeze(0).float().to(device)
target_mask = data['seg'][0].unsqueeze(0).float().to(device)
enc_nf =  [16, 32, 32, 32]
dec_nf =  [32, 32, 32, 32, 32, 16, 16]
in_shape = data['image1'][0].shape[-3:]

#/home/franavarro25/TUM/Master_thesis/voxelmorph_monai/models/multimodal_mi/lambda_1/0200.pt
#'/home/franavarro25/TUM/Master_thesis/voxelmorph_monai/models/multimodal_cobi_contra_patches_local/lambda_10_BEST/0200.pt'
model_dir = '/home/franavarro25/TUM/Master_thesis/voxelmorph_monai/models/multimodal_cobi_contra_patches_local/lambda_10_BEST/0200.pt'
model = vxm.networks.VxmDense.load(model_dir, device).to(device)

transform_model = vxm.layers.SpatialTransformer(in_shape, mode='nearest').to(device)

total_dsc = []
for i, data in enumerate(val_loader):
    with torch.no_grad():
        print('iiiiiiiiiiiiiiiiiiiii', i)
        source = data['image2'][1].unsqueeze(0).float().to(device)#.cuda()
        target = data['image1'][0].unsqueeze(0).float().to(device)#.cuda()
        source_mask_ = data['seg'][1].unsqueeze(0).float().to(device)#.cuda()
        target_mask_ = data['seg'][0].unsqueeze(0).float().to(device)#.cuda()
        
        '''print(torch.unique(source_mask), torch.unique(target_mask))
        print(source_mask.shape, target_mask.shape)
        print(data['image1_meta_dict']['filename_or_obj'],'\n\n', 
              data['image2_meta_dict']['filename_or_obj'],'\n\n',
              data['seg_meta_dict']['filename_or_obj'])'''

        y_source, flow = model(source, target, registration=True)
        y_source_mask = transform_model(source_mask_, flow)

        source_mask = {'seg': y_source_mask}
        target_mask = {'seg': target_mask_}
        
        # binarize mask (20x36x160x192x224)
        source_mask = brainOneHotd(source_mask)['seg'].unsqueeze(0)
        target_mask = brainOneHotd(target_mask)['seg'].unsqueeze(0)
        print(source_mask.shape, target_mask.shape)
        dice_score_labels = monai.metrics.meandice.compute_dice(y_pred=source_mask, y=target_mask, include_background=False)
        print(dice_score_labels, dice_score_labels.mean())
        total_dsc.append(dice_score_labels)
        
        '''loaded_ysource = sitk.GetImageFromArray(y_source.detach().cpu().numpy())
        sitk.WriteImage(loaded_ysource, '/home/franavarro25/Desktop/registered_images/'+str(i)+'_ysource.nii.gz')
        loaded_source = sitk.GetImageFromArray(source.detach().cpu().numpy())
        sitk.WriteImage(loaded_source, '/home/franavarro25/Desktop/registered_images/'+str(i)+'_source.nii.gz')
        loaded_target = sitk.GetImageFromArray(target.detach().cpu().numpy())
        sitk.WriteImage(loaded_target, '/home/franavarro25/Desktop/registered_images/'+str(i)+'_target.nii.gz')
        loaded_ysource_mask = sitk.GetImageFromArray(y_source_mask.detach().cpu().numpy())
        sitk.WriteImage(loaded_ysource_mask, '/home/franavarro25/Desktop/registered_images/'+str(i)+'_ysource_mask.nii.gz')
        loaded_source_mask = sitk.GetImageFromArray(source_mask_.detach().cpu().numpy())
        sitk.WriteImage(loaded_source_mask, '/home/franavarro25/Desktop/registered_images/'+str(i)+'_source_mask.nii.gz')
        loaded_target_mask = sitk.GetImageFromArray(target_mask_.detach().cpu().numpy())
        sitk.WriteImage(loaded_target_mask, '/home/franavarro25/Desktop/registered_images/'+str(i)+'_target_mask.nii.gz')'''


        in_feat_shape_x = source.shape[-3:]
        in_feat_shape_y = target.shape[-3:]
        x = y_source 
        y = target
        mid_slices_moved = [np.take(x.detach().cpu().numpy().squeeze(0), in_feat_shape_x[d]//2, axis=d+1) for d in range(3)]
        mid_slices_moved[1] = np.rot90(mid_slices_moved[1], 1)
        mid_slices_moved[2] = np.rot90(mid_slices_moved[2], -1)
        #print(mid_slices_moved[0].shape, mid_slices_moved[1].shape, mid_slices_moved[2].shape)

        mid_slices_fixed = [np.take(y.detach().cpu().numpy().squeeze(0), in_feat_shape_y[d]//2, axis=d+1) for d in range(3)]
        mid_slices_fixed[1] = np.rot90(mid_slices_fixed[1], 1)
        mid_slices_fixed[2] = np.rot90(mid_slices_fixed[2], -1)
        #print(mid_slices_fixed[0].shape, mid_slices_fixed[1].shape, mid_slices_fixed[2].shape)

        color_map = 'gray'
        # plot the feature map
        plt.figure(figsize=(8,8))
        plt.imshow(mid_slices_moved[0][0], cmap=color_map)
        plt.axis('off')
        plt.title('moved')

        plt.figure(figsize=(8,8))
        plt.imshow(mid_slices_fixed[0][0], cmap=color_map)
        plt.axis('off')
        plt.title('fixed')

        #plt.show()

        # plot the feature map
        plt.figure(figsize=(8,8))
        plt.imshow(mid_slices_moved[1][:,0,:], cmap=color_map)
        plt.axis('off')
        plt.title('moved')

        plt.figure(figsize=(8,8))
        plt.imshow(mid_slices_fixed[1][:,0,:], cmap=color_map)
        plt.axis('off')
        plt.title('fixed')

        #plt.show()

        # plot the feature map
        plt.figure(figsize=(8,8))
        plt.imshow(mid_slices_moved[2][:,0,:], cmap=color_map)
        plt.axis('off')
        plt.title('moved')

        plt.figure(figsize=(8,8))
        plt.imshow(mid_slices_fixed[2][:,0,:], cmap=color_map)
        plt.axis('off')
        plt.title('fixed')

        plt.show()

total_dsc = torch.cat(total_dsc).mean()
print(total_dsc)