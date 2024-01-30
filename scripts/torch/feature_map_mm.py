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
from data_utils import load_data_multimodal, data_loader_multimodal
import matplotlib.pyplot as plt
import numpy as np

# code to get y_source and target
data_dir = '/home/franavarro25/TUM/Master_thesis/dataset/ADNI_remote/ADNI_T1_MNIspace/'
dataset = load_data_multimodal(data_dir=data_dir, csv_file = '/home/franavarro25/TUM/Master_thesis/ANTs_cpp/non_valid_uids.csv') 
training_loader, val_loader = data_loader_multimodal(dataset = dataset, batch_size=1) 
inshape = next(iter(training_loader))['image1'].shape[-3:]
print('inshape', inshape)

device = 'cuda'
data = next(iter(training_loader))
source = data['image2'][0].unsqueeze(0).float().to(device)
target = data['image1'][0].unsqueeze(0).float().to(device)
source_mask = data['seg'][1].unsqueeze(0).float().to(device)
target_mask = data['seg'][0].unsqueeze(0).float().to(device)

model_dir = '/home/franavarro25/TUM/Master_thesis/voxelmorph_monai/models/monomodal_crop96_p8_cobi_patches_local/lambda_1/0000.pt'
model = vxm.networks.VxmDense.load(model_dir, device).cuda()
transformer = vxm.layers.SpatialTransformer(inshape).cuda()

#y_source, flow = model(source, target, registration=True)

# load feature extractor
'''
resnet = vxm.networks.SingleResNet(in_channels=1, resnet_version='base', n_basefilters=16, n_blocks=2).cuda()
for param in resnet.parameters():
    torch.nn.init.normal_(param.data)
    param.requieres_grad = False
resnet.eval()
'''

resnet = vxm.networks.SingleResNet(in_channels=1, resnet_version='base', n_basefilters=16, n_blocks=3)
resnet_dict = torch.load('./pretrained_model/ResNet_pretrained_on_ADNI+NIFD.ckpt')['state_dict']
new_state_dict = {}
for key, value in resnet_dict.items():
    if 'net.' in key:
        key = key.replace('net.', '')
        key = key.replace('res','resnet.')
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

'''
resnet = vxm.networks.SingleResNet(in_channels=1, resnet_version='base', n_basefilters=16, n_blocks=6)
resnet.load_state_dict(torch.load('./pretrained_model/pretrained_ukb_state_dict.pt'))
resnet.cuda()
resnet.eval()
'''

resnet2 = vxm.networks.SingleResNet(in_channels=1, resnet_version='base', n_basefilters=16, n_blocks=3)
resnet2.load_state_dict(torch.load('./pretrained_model/CoBi/t2_l3/0100.pt')['model_state_dict'])
resnet2.cuda()
resnet2.eval()
resnet.cuda()
resnet.eval()

'''
resnet = vxm.networks.SingleResNet(in_channels=1, resnet_version='base', n_basefilters=16, n_blocks=6)
resnet.load_state_dict(torch.load('./pretrained_model/pretrained_ukb_state_dict.pt'))
resnet.cuda()
for param in resnet.parameters():s
    param.data.sub_(torch.mean(param)).div_(torch.std(param)) 
'''

for data in training_loader:
    with torch.no_grad():
        source = data['image2'][0].unsqueeze(0).float().to(device)
        target = data['image1'][0].unsqueeze(0).float().to(device)
        source2 = data['seg'][1].unsqueeze(0).float().to(device)
        print('source', data['image1_meta_dict']['filename_or_obj'])
        print('source', data['image2_meta_dict']['filename_or_obj'])

        #y_source, flow = model(source, target, registration=True)
        #y_source = transformer(source2, flow)
        # get the feature maps
        x, features_x = resnet2(source, get_features=True, get_all_features=True)
        y, features_y = resnet(target, get_features=True, get_all_features=True)

        # see the feature shapes
        for i, feature in enumerate(features_x):
            print('feature:',i,'   ',feature.shape)

        # select the layer in the resnet
        layer = 3
        
        # center the data
        x_mu = features_x[layer].mean(dim=(0,2,3,4), keepdim=True)
        y_mu = features_y[layer].mean(dim=(0,2,3,4), keepdim=True)
        x = features_x[layer] #-y_mu
        y = features_y[layer] #-y_mu

        #L2 normalization
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        y = torch.nn.functional.normalize(y, p=2, dim=1)
        
        print('features shape', features_x[layer].shape)
        print('features shape', features_x[layer].shape)
        in_feat_shape_x = features_x[layer].shape[-3:]
        in_feat_shape_y = features_x[layer].shape[-3:]

        mid_slices_moved = [np.take(x.detach().cpu().numpy().squeeze(), in_feat_shape_x[d]//2, axis=d+1) for d in range(3)]
        mid_slices_moved[1] = np.rot90(mid_slices_moved[1], 1)
        mid_slices_moved[2] = np.rot90(mid_slices_moved[2], -1)
        print(mid_slices_moved[0].shape, mid_slices_moved[1].shape, mid_slices_moved[2].shape)

        mid_slices_fixed = [np.take(y.detach().cpu().numpy().squeeze(), in_feat_shape_y[d]//2, axis=d+1) for d in range(3)]
        mid_slices_fixed[1] = np.rot90(mid_slices_fixed[1], 1)
        mid_slices_fixed[2] = np.rot90(mid_slices_fixed[2], -1)
        print(mid_slices_fixed[0].shape, mid_slices_fixed[1].shape, mid_slices_fixed[2].shape)


        # plot the feature map
        plt.figure(figsize=(8,8))
        for idx in range(16):
            #mean = np.mean(mid_slices_moved[0][idx])
            #var = np.mean((mid_slices_moved[0][idx]-mean)**2)
            #print('var', idx,var)
            plt.subplot(4,4,idx+1)
            plt.imshow(mid_slices_moved[0][idx], cmap='viridis')
            plt.axis('off')
            plt.title('moved')
        plt.figure(figsize=(8,8))
        for idx in range(16):
            #mean = np.mean(mid_slices_fixed[0][idx])
            #var = np.mean((mid_slices_fixed[0][idx]-mean)**2)
            #print('var', idx,var)
            plt.subplot(4,4,idx+1)
            plt.imshow(mid_slices_fixed[0][idx], cmap='viridis')
            plt.axis('off')
            plt.title('fixed')

        plt.figure(figsize=(8,8))
        for idx in range(16):
            #mean = np.mean(mid_slices_moved[0][idx])
            #var = np.mean((mid_slices_moved[0][idx]-mean)**2)
            #print('var', idx,var)
            plt.subplot(4,4,idx+1)
            plt.imshow(mid_slices_moved[0][idx,22:30,24:32], cmap='viridis')
            plt.axis('off')
            plt.title('moved')
        plt.figure(figsize=(8,8))
        for idx in range(16):
            #mean = np.mean(mid_slices_fixed[0][idx])
            #var = np.mean((mid_slices_fixed[0][idx]-mean)**2)
            #print('var', idx,var)
            plt.subplot(4,4,idx+1)
            plt.imshow(mid_slices_fixed[0][idx,22:30,24:32], cmap='viridis')
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
            plt.axis('off')
            plt.title('moved')

        plt.figure(figsize=(8,8))
        for idx in range(16):
            #mean = np.mean(mid_slices_fixed[1][:,idx,:])
            #var = np.mean((mid_slices_fixed[1][:,idx,:]-mean)**2)
            #print('var', idx, var)
            plt.subplot(4,4,idx+1)
            plt.imshow(mid_slices_fixed[1][:,idx,:], cmap='viridis')
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
            plt.axis('off')
            plt.title('moved')

        plt.figure(figsize=(8,8))
        for idx in range(16):
            #mean = np.mean(mid_slices_fixed[2][:,idx,:])
            #var = np.mean((mid_slices_fixed[2][:,idx,:]-mean)**2)
            #print('var', idx, var)
            plt.subplot(4,4,idx+1)
            plt.imshow(mid_slices_fixed[2][:,idx,:], cmap='viridis')
            plt.axis('off')
            plt.title('fixed')

        plt.show()

        del source, target, mid_slices_fixed, x, y, features_x, features_y