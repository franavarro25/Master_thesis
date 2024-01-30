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
from utils import *
import matplotlib.pyplot as plt
import numpy as np

# code to get y_source and target
dataset, transformations = load_data('/home/franavarro25/TUM/Master_thesis/dataset/OASIS/train/**/') 
train_loader, val_loader = data_loader(dataset=dataset, batch_size=1) 
data = next(iter(train_loader))
device = 'cuda'
source = data['image'][0].unsqueeze(0).float().to(device)
target = data['image'][1].unsqueeze(0).float().to(device)
source_mask = data['label'][0].unsqueeze(0).float().to(device)
target_mask = data['label'][1].unsqueeze(0).float().to(device)
enc_nf =  [16, 32, 32, 32]
dec_nf =  [32, 32, 32, 32, 32, 16, 16]
in_shape = next(iter(train_loader))['image'].shape[-3:]
'''model = vxm.networks.VxmDense(
            inshape=in_shape,
            nb_unet_features=[enc_nf, dec_nf],
            int_steps=7,
            int_downsize=2)'''
model_dir = '/home/franavarro25/TUM/Master_thesis/voxelmorph_monai/models/presentation/ncc/lambda_1/0500.pt'
model = vxm.networks.VxmDense.load(model_dir, device).to(device)

y_source, flow = model(source, target, registration=True)

# load feature extractor
'''
resnet = vxm.networks.SingleResNet(in_channels=1, resnet_version='base', n_basefilters=16, n_blocks=2).cuda()
for param in resnet.parameters():
    torch.nn.init.normal_(param.data)
    param.requieres_grad = False
resnet.eval()
'''

resnet = vxm.networks.SingleResNet(in_channels=1, resnet_version='base', n_basefilters=16, n_blocks=3)
resnet_dict = torch.load('/home/franavarro25/TUM/Master_thesis/voxelmorph_monai/pretrained_model/ResNet_pretrained_on_ADNI+NIFD.ckpt')['state_dict']
new_state_dict = {}
for key, value in resnet_dict.items():
    if 'net.' in key:
        key = key.replace('net.', '')
        key = key.replace('res','resnet.')
    #if 'resnet.blocks.2.' in key:
     #   continue
    if 'resnet.blocks.3.' in key:
        continue
    if 'resnet.blocks.4.' in key:
        continue
    if 'resnet.blocks.5.' in key:
        continue
    if key in resnet.state_dict():
        new_state_dict[key] = value
resnet_dict['state_dict'] = new_state_dict
resnet.load_state_dict(resnet_dict['state_dict'])
del resnet_dict, new_state_dict
resnet.cuda()
resnet.eval()
for param in resnet.parameters():
    param.requires_grad = False


for data in val_loader:
    with torch.no_grad():
        source = data['image'][0].unsqueeze(0).float().to(device)
        target = data['image'][1].unsqueeze(0).float().to(device)
        y_source, flow = model(source, target, registration=True)
        # get the feature maps
        x, features_x = resnet(source, get_features=True, get_all_features=True)
        y, features_y = resnet(target, get_features=True, get_all_features=True)

        # see the feature shapes
        for i, feature in enumerate(features_x):
            print('feature:',i,'   ',feature.shape)

        # select the layer in the resnet
        layer = 2

        # center the data
        x_mu = features_x[layer].mean(dim=(0,2,3,4), keepdim=True)
        y_mu = features_y[layer].mean(dim=(0,2,3,4), keepdim=True)
        x = features_x[layer] - y_mu
        y = features_y[layer] - y_mu
        
        #L2 normalization
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        y = torch.nn.functional.normalize(y, p=2, dim=1)
         
        #print('check normalization', torch.sum(x.abs(),dim=1), torch.sum(y.abs(),dim=1))
        #print('check normalization', torch.sum(x * x, dim=1), torch.sum(y * y, dim=1))

        patch_size = 8 
        x = divide_volume(x,(patch_size, patch_size, patch_size))[1010]
        y = divide_volume(y,(patch_size, patch_size, patch_size))[1010]
        print('Are featues x and y equal?', torch.equal(x,y))

        print('features shape', x.shape)
        print('features shape', y.shape)
        in_feat_shape_x = x.shape[-3:]
        in_feat_shape_y = x.shape[-3:]

        ###mid_slices_moved = [np.take(x.squeeze(), in_feat_shape_x[d]//2, axis=d+1) for d in range(3)]
        mid_slices_moved = [np.take(x.squeeze(), 0, axis=d+1) for d in range(3)]
        mid_slices_moved[1] = np.rot90(mid_slices_moved[1], 1)
        mid_slices_moved[2] = np.rot90(mid_slices_moved[2], -1)
        print(mid_slices_moved[0].shape, mid_slices_moved[1].shape, mid_slices_moved[2].shape)

        ###mid_slices_fixed = [np.take(y.squeeze(), in_feat_shape_y[d]//2, axis=d+1) for d in range(3)]
        mid_slices_fixed = [np.take(y.squeeze(), 0, axis=d+1) for d in range(3)]
        mid_slices_fixed[1] = np.rot90(mid_slices_fixed[1], 1)
        mid_slices_fixed[2] = np.rot90(mid_slices_fixed[2], -1)
        print(mid_slices_fixed[0].shape, mid_slices_fixed[1].shape, mid_slices_fixed[2].shape)


        # plot the feature map
        plt.figure(figsize=(8,8))
        for idx in range(16):
            print('moved', idx, mid_slices_moved[0][idx])
            plt.subplot(4,4,idx+1)
            plt.imshow(mid_slices_moved[0][idx], cmap='viridis')
            plt.colorbar()
            plt.axis('off')
            plt.title('moved')
        plt.figure(figsize=(8,8))
        for idx in range(16):
            print('fixed', idx, mid_slices_fixed[0][idx])
            plt.subplot(4,4,idx+1)
            plt.imshow(mid_slices_fixed[0][idx], cmap='viridis')
            plt.colorbar()
            plt.axis('off')
            plt.title('fixed')

        #plt.show()

        # plot the feature map
        plt.figure(figsize=(8,8))
        for idx in range(16):
            #mean = np.mean(mid_slices_moved[1][:,idx,:])
            #var = np.mean((mid_slices_moved[1][:,idx,:]-mean)**2)
            #print('var', idx, var)
            plt.subplot(4,4,idx+1)
            plt.imshow(mid_slices_moved[1][:,idx,:], cmap='viridis')
            plt.colorbar()
            plt.axis('off')
            plt.title('moved')

        plt.figure(figsize=(8,8))
        for idx in range(16):
            #mean = np.mean(mid_slices_fixed[1][:,idx,:])
            #var = np.mean((mid_slices_fixed[1][:,idx,:]-mean)**2)
            #print('var', idx, var)
            plt.subplot(4,4,idx+1)
            plt.imshow(mid_slices_fixed[1][:,idx,:], cmap='viridis')
            plt.colorbar()
            plt.axis('off')
            plt.title('fixed')

        #plt.show()

        # plot the feature map
        plt.figure(figsize=(8,8))
        for idx in range(16):
            #mean = np.mean(mid_slices_moved[2][:,idx,:])
            #var = np.mean((mid_slices_moved[2][:,idx,:]-mean)**2)
            #print('var', idx, var)
            plt.subplot(4,4,idx+1)
            plt.imshow(mid_slices_moved[2][:,idx,:], cmap='viridis')
            plt.colorbar()
            plt.axis('off')
            plt.title('moved')

        plt.figure(figsize=(8,8))
        for idx in range(16):
            #mean = np.mean(mid_slices_fixed[2][:,idx,:])
            #var = np.mean((mid_slices_fixed[2][:,idx,:]-mean)**2)
            #print('var', idx, var)
            plt.subplot(4,4,idx+1)
            plt.imshow(mid_slices_fixed[2][:,idx,:], cmap='viridis')
            plt.colorbar()
            plt.axis('off')
            plt.title('fixed')

        plt.show()

        del source, target, y_source, flow, mid_slices_fixed, x, y, features_x, features_y