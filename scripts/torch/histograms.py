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

model_dir = '/home/franavarro25/TUM/Master_thesis/voxelmorph_monai/models/monomodal_crop96_p8_cobi_patches_local/lambda_1/0200.pt'
model = vxm.networks.VxmDense.load(model_dir, device).to(device)

transform_model = vxm.layers.SpatialTransformer(in_shape, mode='nearest').to(device)

ae1 = vxm.networks.AutoEncoder().to(device)
ae1.load_state_dict(torch.load('./pretrained_model/ae1/NCC/0010.pt')['model_state_dict'])
ae1.eval()

ae2 = vxm.networks.AutoEncoder().to(device)
ae2.load_state_dict(torch.load('./pretrained_model/ae2/NCC/0010.pt')['model_state_dict'])
ae2.eval()
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

for param in resnet.parameters():
    torch.nn.init.normal_(param.data, mean=0.0, std=10.0)
    param.requires_grad = False
resnet.eval()'''


int_meanstd, feat_meanstd = [],[]
for i, data in enumerate(val_loader):
    with torch.no_grad():
        #if i == 0:
            print('iiiiiiiiiiiiiiiiiiiii', i)
            source = data['image2'][1].unsqueeze(0).float().to(device)
            target = data['image1'][0].unsqueeze(0).float().to(device)
            source_mask_ = data['seg'][1].unsqueeze(0).float().to(device)
            target_mask_ = data['seg'][0].unsqueeze(0).float().to(device)
            
            y_source, flow = model(source, target, registration=True)
            y_source_mask = transform_model(source_mask_, flow)

            x0, features_x0 = ae2(source )
            x, features_x = ae2(y_source )
            y, features_y = ae1(target )

            x0 = features_x0[0]
            x = features_x[0]
            y = features_y[0]
            
            x0 = torch.nn.functional.normalize(x0, p=2, dim=1)
            x = torch.nn.functional.normalize(x, p=2, dim=1)
            y = torch.nn.functional.normalize(y, p=2, dim=1)
            del features_x0, features_x, features_y

            int_mean = (source.mean()-target.mean()).abs()
            int_std = (source.std()-target.std()).abs()
            int_meanstd.append([int_mean, int_std])
            
            feat_mean = (x.mean()-y.mean()).abs()
            feat_std = (x.std()-y.std()).abs()
            feat_meanstd.append([feat_mean, feat_std])
            print('targett1', target.mean(), target.std())
            print('sourcet2', source.mean(), target.std())
            print('source-target', int_mean, int_std)
            print('y_source_feats-target_feats', feat_mean, feat_std)

            source = source.detach().numpy()
            target = target.detach().numpy()
            y_source = y_source.detach().numpy()
            x = x.detach().numpy()
            y = y.detach().numpy()
            x0 = x0.detach().numpy()

            plt.figure(figsize=(18,6))
            fig,ax=plt.subplots(2,3)
            _=ax[0,0].hist(source[source>0],bins=64)
            _=ax[0,1].hist(target[target>0],bins=64)
            _=ax[0,2].hist(y_source[y_source>0],bins=64)
            _=ax[1,0].hist(x0[x0>0],bins=64)
            _=ax[1,1].hist(y[y>0],bins=64)
            _=ax[1,2].hist(x[x>0],bins=64)
            plt.show()

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
        #else: break

print(len(int_meanstd), len(int_meanstd[0]))
print(len(feat_meanstd), len(feat_meanstd[0]))

print(np.mean(int_meanstd, axis=0))
print(np.mean(feat_meanstd, axis=0))