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
from utils import *
from losses import contextual_bilateral_loss3D_local
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

# load and prepare training data
# dict with keys:image,label;values:file_names
dataset = load_data_multimodal(data_dir='/home/franavarro25/TUM/Master_thesis/dataset/ADNI_remote/ADNI_T1_MNIspace/', 
                               csv_file = '/home/franavarro25/TUM/Master_thesis/ANTs_cpp/non_valid_uids.csv') 

# dataloader--> dict with keys: image,labels,image_metadata,label_metadata; values: real 3d volumes 
training_loader, val_loader = data_loader_multimodal(dataset = dataset, 
                                                     batch_size=1) 
del val_loader
iterat = iter(training_loader)
data = next(iterat)
data = next(iterat)

# Load feature extractor
resnet = vxm.networks.SingleResNet(in_channels=1, resnet_version='base', n_basefilters=16, n_blocks=3)
resnet_dict = torch.load('/home/franavarro25/TUM/Master_thesis/voxelmorph_monai/pretrained_model/ResNet_pretrained_on_ADNI+NIFD.ckpt')['state_dict']
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
for param in resnet.parameters():
    #nn.init.normal_(param.data)
    param.requires_grad = False
resnet.cuda()
resnet.eval()


# load the model 
model_dir = '/home/franavarro25/TUM/Master_thesis/voxelmorph_monai/models/monomodal_crop96_p8_cobi_patches_local/lambda_1/0200.pt'
device = 'cuda'
model = vxm.networks.VxmDense.load(model_dir, device).cuda()

# registration
print('data',data.keys())
source = data['image1'][1].unsqueeze(0).float().to(device)
target = data['image1'][0].unsqueeze(0).float().to(device)
print('source', data['image1_meta_dict']['filename_or_obj'])
model.eval()
y_source, flow = model(source, target, registration=True)

# extract the features
x, features_x = resnet(y_source, get_features=True, get_all_features=True)
y, features_y = resnet(target, get_features=True, get_all_features=True)
x = features_x[0]
y = features_y[0]

# see if the targets are all the same\n",
print(torch.abs(x-y).mean())
print('are x and y the same?', torch.equal(x,y))

# the distribution of x and y
print('x_mean',x.mean().item(), 'x_std',x.std().item(), 'y_mean',y.mean().item(),'y_std',y.std().item())

# see if the targets are all the same
print(torch.abs(x-y).mean())
print('are x and y the same?', torch.equal(x,y))

# the distribution of x and y
print('x_mean',x.mean().item(), 'x_std',x.std().item(), 'y_mean',y.mean().item(),'y_std',y.std().item())

#y = target
print(x.shape, y.shape, features_x[2].shape)

# %%
# center the data
x_mu = x.mean(dim=(0,2,3,4), keepdim=True)
y_mu = y.mean(dim=(0,2,3,4), keepdim=True)
x = x -y_mu
y = y -y_mu

#L2 normalization
x = F.normalize(x, p=2, dim=1)
y = F.normalize(y, p=2, dim=1)

# %%
print(x.shape, y.shape)

# %%
# get patches
patch_size = 8

x_patches = divide_volume(volume=x, patch_size=(patch_size,patch_size,patch_size))
y_patches = divide_volume(volume=y, patch_size=(patch_size,patch_size,patch_size))
x_patches_mri = divide_volume(volume=source, patch_size=(patch_size,patch_size,patch_size))
y_patches_mri = divide_volume(volume=target, patch_size=(patch_size,patch_size,patch_size))
print('los patches tienen shape', x.shape, y.shape)
del features_x, features_y, x, y, y_source, flow, model, resnet 

# %%
max_tot = []
# add spatial information
grid = compute_meshgrid_3d((1,1,patch_size,patch_size,patch_size)).to('cuda')
sp_dist = torch.exp(-compute_l2_distance_3d(grid, grid))

for i, (x_patch, y_patch) in enumerate(zip(x_patches, y_patches)):
    dist_raw = compute_cosine_distance_3d(x_patch, y_patch)
    dist_raw = ((1)*dist_raw + 1*sp_dist)/2
    dist_tilde = compute_relative_distance(1-dist_raw) # eq(2) normalizing distances
    cx = compute_cx(dist_tilde, 0.1) # eq(3,4) distance-->similarit + invariant
    max_tot.append(torch.max(cx, dim=1)[0])
max_cx = torch.stack(max_tot, dim=0)
cx_tot = torch.mean(max_cx,dim=2)  # Eq(1) shape--> n_patches x batch
cx_loss = torch.mean(-torch.log(cx_tot.mean() + 1e-5)) # Eq(5)
print(cx_loss)

# %%
# compute cosine distance
x = x_patches[1010]
y = y_patches[1010]
x_mri = x_patches_mri[1010]
y_mri = y_patches_mri[1010]
d_cos = compute_cosine_distance_3d(x,y) ###sim
print(d_cos.min(), d_cos.max()) # min value in d[0,11172,6860]

# compute ncc
ncc = vxm.losses.NCC().loss
#d_ncc = ncc(x,y)

# d combined !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
sim_combined = (1*d_cos + 1*sp_dist)/2
d = 1-sim_combined
#print(d.squeeze().shape)
print(d.min(),d.max())

# %%
'''# to just see a slice 
# extract the features
x, features_x = resnet(target, get_features=True, get_all_features=True)
y, features_y = resnet(target, get_features=True, get_all_features=True)
x = features_x[4]
y = features_y[4]
print(x.shape,y.shape)
d_mse = compute_l2_distance_3d(x,y)

# center the data
x_mu = x.mean(dim=(0,2,3,4), keepdim=True)
y_mu = y.mean(dim=(0,2,3,4), keepdim=True)
x = x - y_mu
y = y - y_mu

# L2 normalization
x = F.normalize(x, p=2, dim=1)
y = F.normalize(y, p=2, dim=1)

# take the middle slice
x = x[:,:,:,:,x.shape[4]//2]
y = y[:,:,:,:,y.shape[4]//2]
print(x.shape, y.shape)

# squeeze
x = x.reshape(1, 64, -1)  # (N, C, H*W)
y = y.reshape(1, 64, -1)
print(x.shape, y.shape)

# compute cosine similarity
cosine_sim = torch.bmm(x.transpose(1, 2),y)

# compute the cosine distance
d = (1 - cosine_sim)

print(d.shape)'''


mid_slices_moved_mri = [np.take(x_mri.detach().cpu().numpy().squeeze(0), 8//2, axis=d+1) for d in range(3)]
mid_slices_moved_mri[1] = np.rot90(mid_slices_moved_mri[1], 1)
mid_slices_moved_mri[2] = np.rot90(mid_slices_moved_mri[2], -1)
print(mid_slices_moved_mri[0].shape, mid_slices_moved_mri[1].shape, mid_slices_moved_mri[2].shape)

mid_slices_fixed_mri = [np.take(y_mri.detach().cpu().numpy().squeeze(0), 8//2, axis=d+1) for d in range(3)]
mid_slices_fixed_mri[1] = np.rot90(mid_slices_fixed_mri[1], 1)
mid_slices_fixed_mri[2] = np.rot90(mid_slices_fixed_mri[2], -1)
print(mid_slices_fixed_mri[0].shape, mid_slices_fixed_mri[1].shape, mid_slices_fixed_mri[2].shape)

mid_slices_moved = [np.take(x.detach().cpu().numpy().squeeze(), 8//2, axis=d+1) for d in range(3)]
mid_slices_moved[1] = np.rot90(mid_slices_moved[1], 1)
mid_slices_moved[2] = np.rot90(mid_slices_moved[2], -1)
print(mid_slices_moved[0].shape, mid_slices_moved[1].shape, mid_slices_moved[2].shape)

mid_slices_fixed = [np.take(y.detach().cpu().numpy().squeeze(), 8//2, axis=d+1) for d in range(3)]
mid_slices_fixed[1] = np.rot90(mid_slices_fixed[1], 1)
mid_slices_fixed[2] = np.rot90(mid_slices_fixed[2], -1)
print(mid_slices_fixed[0].shape, mid_slices_fixed[1].shape, mid_slices_fixed[2].shape)

# plot the feature map
plt.figure(figsize=(4,4))
plt.imshow(mid_slices_moved_mri[0][0], cmap='gray')
plt.axis('off')
plt.title('moved')

plt.figure(figsize=(4,4))
plt.imshow(mid_slices_fixed_mri[0][0], cmap='gray')
plt.axis('off')
plt.title('fixed')


# plot the feature map
plt.figure(figsize=(4,4))
for idx in range(16):
    #mean = np.mean(mid_slices_moved[0][idx])
    #var = np.mean((mid_slices_moved[0][idx]-mean)**2)
    #print('var', idx,var)
    plt.subplot(4,4,idx+1)
    plt.imshow(mid_slices_moved[0][idx], cmap='viridis')
    plt.axis('off')
    plt.title('moved')

plt.figure(figsize=(4,4))
for idx in range(16):
    #mean = np.mean(mid_slices_fixed[0][idx])
    #var = np.mean((mid_slices_fixed[0][idx]-mean)**2)
    #print('var', idx,var)
    plt.subplot(4,4,idx+1)
    plt.imshow(mid_slices_fixed[0][idx], cmap='viridis')
    plt.axis('off')
    plt.title('fixed')
plt.show()

# %%
# plot cosine distance map
plt.imshow(d.squeeze().detach().cpu().numpy())
plt.colorbar()
plt.title('Distance Map')
plt.ylabel('x_i')
plt.xlabel('y_j')
plt.show()

# %%
# eq2, compute relative distance = d/(d.min(j)+1e-5)
print(torch.min(d,dim=2)[0], '\n', torch.min(d,dim=2)[0]+1e-5 ,'\n' ,torch.max(d,dim=(2))[0])
dist_tilde = compute_relative_distance(d)
print(dist_tilde.min(), dist_tilde.max())
print('dist_tilde', dist_tilde)

# %%
# plot normalized distance map
plt.imshow(dist_tilde.squeeze().detach().cpu().numpy())
plt.colorbar()
plt.title('Normalized Distance Map')
plt.ylabel('x_i')
plt.xlabel('y_j')
plt.show()
# %%
# eq3 
band_width = 0.1
w = torch.exp((1 - dist_tilde) / band_width)
print(w.min(), w.max())
print('w', w)

# %%
# plot similarity map before computing mean(max(cx))
plt.imshow(w.squeeze().detach().cpu().numpy())
plt.colorbar()
plt.title('Similarity Map')
plt.ylabel('x_i')
plt.xlabel('y_j')
plt.show()

# plot cosine distance map
plt.imshow(d.squeeze().detach().cpu().numpy())
plt.colorbar()
plt.title('Distance Map')
plt.ylabel('x_i')
plt.xlabel('y_j')
plt.show()

# %%
# eq4
cx = w / torch.sum(w, dim=2, keepdim=True)
plt.imshow(cx.squeeze().detach().cpu().numpy())
plt.colorbar()
plt.title('Affinity matrix Cxij')
plt.ylabel('x_i')
plt.xlabel('y_j')
plt.show()
print('cx', cx)
print('check if all elements in the diagonal are 1', torch.all(cx == torch.diag(torch.diag(cx.squeeze()))))

# CX
CX = torch.mean(torch.max(cx, axis=1)[0])
print('CX similarity', CX)

# %%
# plot similarity map before computing mean(max(cx))
plt.imshow(cx[0,200:250,200:250].squeeze().detach().cpu().numpy())
plt.colorbar()
plt.title('Affinity matrix Cxij small')
plt.ylabel('x_i')
plt.xlabel('y_j')
plt.show()


