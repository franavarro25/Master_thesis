import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import SimpleITK as sitk

### my imports
import os
import time
# import voxelmorph with pytorch backend from source
os.environ['NEURITE_BACKEND'] = 'pytorch'  ###############333
os.environ['VXM_BACKEND'] = 'pytorch'      ############
import sys 
sys.path.append('/home/franavarro25/TUM/Master_thesis/voxelmorph_monai')
import voxelmorph as vxm

from data_utils import load_data_multimodal, data_loader_resnet
from vxm_network import vxm_model
from losses import Contrastive_loss_resnet, Contrastive_loss_resnet_patches, CXB_resnset
import wandb 


# load and prepare training data
# dict with keys:image,label;values:file_names
data_dir = '/home/franavarro25/TUM/Master_thesis/dataset/ADNI_remote/ADNI_T1_MNIspace/'
dataset = load_data_multimodal(data_dir=data_dir, csv_file = '/home/franavarro25/TUM/Master_thesis/ANTs_cpp/non_valid_uids.csv') 


# dataloader--> dict with keys: image,labels,image_metadata,label_metadata; values: real 3d volumes 
training_loader, val_loader = data_loader_resnet(dataset = dataset, batch_size=1) 
print(len(training_loader), len(val_loader))

# extract shape from sampled input [2(images),1,160,192,224]]
inshape = next(iter(training_loader))['image2'].shape[-3:]
print('inshape', inshape)


# device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define your model
resnet_t1 = vxm.networks.SingleResNet(in_channels=1, resnet_version='base', n_basefilters=16, n_blocks=3).to(device)
resnet_t2 = vxm.networks.SingleResNet(in_channels=1, resnet_version='base', n_basefilters=16, n_blocks=3).to(device)

# Load RESNET pretrained on T1 images
resnet_t1_dict = torch.load('./pretrained_model/ResNet_pretrained_on_ADNI+NIFD.ckpt')['state_dict']
new_state_dict = {}
# Modify keys to match the model's parameter names
for key, value in resnet_t1_dict.items():
    # Check if the key is in the model's state_dict
    if 'net.' in key:
        key = key.replace('net.', '')
        key = key.replace('res','resnet.')
    if 'resnet.blocks.3.' in key:
        continue
    if 'resnet.blocks.4.' in key:
        continue
    if 'resnet.blocks.5.' in key:
        continue
    if key in resnet_t1.state_dict():
        # Modify key as needed to match the model's parameter names
        new_state_dict[key] = value
resnet_t1_dict['state_dict'] = new_state_dict
resnet_t1.load_state_dict(resnet_t1_dict['state_dict'])
del resnet_t1_dict, new_state_dict

for param in resnet_t1.parameters():
    param.requires_grad = False
resnet_t1.eval()



# See features shape
x, features_x = resnet_t2(next(iter(training_loader))['image2'].to(device), get_features=True, get_all_features=True)
for i, feature in enumerate(features_x):
    print('feature:',i,'   ',feature.shape)


# Define loss function and optimizer
criterion = CXB_resnset()
layer = 3
print('criterion', criterion)
lr = 0.001
#optimizer_t1 = optim.Adam(resnet_t1.parameters(), lr=lr)
optimizer_t2 = optim.Adam(resnet_t2.parameters(), lr=lr)


# Training loop
model_dir = './pretrained_model/CoBi'
os.makedirs(model_dir, exist_ok=True)
#os.makedirs(os.path.join(model_dir,'t1'), exist_ok=True)
os.makedirs(os.path.join(model_dir,'t2'), exist_ok=True)
num_epochs = 30

wandb_bool = False

if wandb_bool:
    run = wandb.init(project='voxelmorph',group= 'pretrain_feature_extractor',
                     name= 'Contextual loss',
    config={
            'number-epochs:':num_epochs, 'lr':lr, 'data_val':len(val_loader),
            'similarity:':'cosine', 'layer':layer, 'patch_size':(8,8,8),
            'data_train':len(training_loader),
            })

for epoch in range(1,num_epochs+1):
    if epoch == 1:
        resnet_t1.eval()
        resnet_t2.eval()
        loss_epoch_eval = []
        epoch_eval_time = []
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                eval_start_time = time.time()
                t2 = data['image2'].float().to(device)
                t1 = data['image1'].float().to(device)
                #print('image2', data['image2_meta_dict']['filename_or_obj'][0])
                #print('image1', data['image1_meta_dict']['filename_or_obj'][0])

                t1, t1_output = resnet_t1(t1, get_features=True, get_all_features=True)
                t2, t2_output = resnet_t2(t2, get_features=True, get_all_features=True)

                loss = criterion(t2_output, t1_output, layer)
                loss_epoch_eval.append(loss.item())

                epoch_eval_time.append(time.time()-eval_start_time)

        avg_eval_time = np.mean(epoch_eval_time)
        avg_loss = np.mean(loss_epoch_eval)
        print(f"Epoch [{epoch-1}/{num_epochs}], Loss(eval): {avg_loss:.4f}, time(eval): {avg_eval_time:.4f}")
        if  wandb_bool:
            wandb.log({'epoch':epoch-1, 'loss(eval)':avg_loss.item(), 'time(eval)':avg_eval_time.item()}, step=epoch)


    if epoch % 10 == 0:
        '''checkpoint_t1 = {
        'model_state_dict': resnet_t1.state_dict(),
        'optimizer_state_dict': optimizer_t1.state_dict(),
        'epoch': epoch,  
        'lr': lr,
        }'''
        checkpoint_t2 = {
        'model_state_dict': resnet_t2.state_dict(),
        'optimizer_state_dict': optimizer_t2.state_dict(),
        'epoch': epoch,  
        'lr': lr,
        }
        #torch.save(checkpoint_t1, os.path.join(model_dir,'t1', '%04d.pt' % epoch))
        torch.save(checkpoint_t2, os.path.join(model_dir,'t2', '%04d.pt' % epoch))

        resnet_t1.eval()
        resnet_t2.eval()
        loss_epoch_eval = []
        epoch_eval_time = []
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                eval_start_time = time.time()
                t2 = data['image2'].float().to(device)
                t1 = data['image1'].float().to(device)

                t1, t1_features = resnet_t1(t1, get_features=True, get_all_features=True)
                t2, t2_features = resnet_t2(t2, get_features=True, get_all_features=True)

                loss = criterion(t2_features, t1_features, layer)
                loss_epoch_eval.append(loss.item())

                epoch_eval_time.append(time.time()-eval_start_time)

        avg_eval_time = np.mean(epoch_eval_time)
        avg_loss = np.mean(loss_epoch_eval)
        print(f"Epoch [{epoch}/{num_epochs}], Loss(eval): {avg_loss:.4f}, time(eval): {avg_eval_time:.4f}")
        if  wandb_bool:
            wandb.log({'epoch':epoch, 'loss(eval)':avg_loss.item(), 'time(eval)':avg_eval_time.item()}, step=epoch)

    #resnet_t1.train()
    resnet_t2.train()
    loss_epoch = []
    epoch_time = []
    for i, data in enumerate(training_loader):
        step_start_time = time.time()
        # ae1--> x=image1; ae2--> x=image2
        t2 = data['image2'].float().to(device)
        t1 = data['image1'].float().to(device)
        #sitk.WriteImage(sitk.GetImageFromArray(x.detach().cpu().numpy()), '/home/franavarro25/Desktop/loaded_images/'+str(i)+'_x.nii.gz')
        #print('data', data['image2_meta_dict']['filename_or_obj'])
        #x_mask = data['seg'].unsqueeze(0).float().to(device)

        # forward pass
        t1, t1_features = resnet_t1(t1, get_features=True, get_all_features=True)
        t2, t2_features = resnet_t2(t2, get_features=True, get_all_features=True)
        
        # loss computation
        loss = criterion(t2_features, t1_features, layer)

        # tracking loss
        loss_epoch.append(loss.item())
        
        # zero gradients, backward pass and optimization
        #optimizer_t1.zero_grad()
        optimizer_t2.zero_grad()
        loss.backward()
        #optimizer_t1.step()
        optimizer_t2.step()

        epoch_time.append(time.time() - step_start_time)

    avg_time = np.mean(epoch_time)
    avg_loss = np.mean(loss_epoch)
    # Print average loss for the epoch
    print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}, time(sec/img): {avg_time:.4f}")
    if  wandb_bool:
        wandb.log({'epoch':epoch, 'loss':avg_loss.item(), 'time(sec/img)':avg_time.item() }, step=epoch)
    # trainking loss


'''checkpoint_t1 = {
        'model_state_dict': resnet_t1.state_dict(),
        'optimizer_state_dict': optimizer_t1.state_dict(),
        'epoch': num_epochs,  
        'lr': lr,
        }'''
checkpoint_t2 = {
        'model_state_dict': resnet_t2.state_dict(),
        'optimizer_state_dict': optimizer_t2.state_dict(),
        'epoch': num_epochs,  
        'lr': lr,
        }
#torch.save(checkpoint_t1, os.path.join(model_dir,'t1', '%04d_final.pt' % num_epochs))
torch.save(checkpoint_t2, os.path.join(model_dir,'t2', '%04d_final.pt' % num_epochs))
print("Training finished.")
