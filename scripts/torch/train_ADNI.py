#!/usr/bin/env python

"""
Example script to train a VoxelMorph model.

You will likely have to customize this script slightly to accommodate your own data. All images
should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed.
Otherwise, registration will be scan-to-scan.

If you use this code, please cite the following, and read function docs for further info/citations.

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration G. Balakrishnan, A.
    Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. IEEE TMI: Transactions on Medical Imaging. 38(8). pp
    1788-1800. 2019. 

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
implied. See the License for the specific language governing permissions and limitations under the
License.
"""
import gc ####################
import os
import random
import argparse
import time
import numpy as np
import torch
import glob
import monai
from monai.data import DataLoader
import wandb

# import voxelmorph with pytorch backend from source
os.environ['NEURITE_BACKEND'] = 'pytorch'  ###############333
os.environ['VXM_BACKEND'] = 'pytorch'      ############
import sys 
sys.path.append('/home/franavarro25/TUM/Master_thesis/voxelmorph_monai')
import voxelmorph as vxm

# import my utils
from data_utils import load_data_multimodal, data_loader_multimodal, BrainOneHotd
from vxm_network import vxm_model
from losses import loss_functions, two_loss_functions
from utils import run_epoch_multimodal, divide_volume
from metrics import SDlogDetJac

import SimpleITK as sitk

# parse the commandline
parser = argparse.ArgumentParser()

# Using wandb
parser.add_argument('--wandb', default=True , type=bool, help='save training values with weights and biases')
parser.add_argument('--name', help='name of the wandb run')

# data organization parameters
parser.add_argument('--data-dir', default='/home/franavarro25/TUM/Master_thesis/dataset/ADNI_remote/ADNI_T1_MNIspace/', 
                    help='line-seperated list of training files')

parser.add_argument('--img-prefix', help='optional input image file prefix')
parser.add_argument('--img-suffix', help='optional input image file suffix')
parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')
parser.add_argument('--model-output',default='0',
                    help='model output directory (default: 0)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')

# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of training epochs (default: 1000)')
parser.add_argument('--steps-per-epoch', type=int, default=100,
                    help='frequency of model saves (default: 100)')
parser.add_argument('--load-model', help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')

# loss hyperparameters
parser.add_argument('--loss-rgb', default='ncc',type=str,
                    help='intensity reconstruction loss - can be l1,l2 (default: l1)')
parser.add_argument('--loss-feat', default='cobi_patches_local_mm',type=str,
                    help='feature reconstruction loss - can be cx or cobi (default: cobi)')
parser.add_argument('--similarity', default='cosine', type=str, 
                    help='feature reconstruction loss - can be(l1,l2,cosine)')
parser.add_argument('--lambda', type=float, dest='weight', default=1,
                    help='weight of deformation loss (default: 0.01)')
                    # best lambdas--> mind=0.1; mi=3; ngf=0.001
parser.add_argument('--weight-sp', default=1, type=float,
                    help='(default=0.1)weight for cx combination (1-weight:features+weight:spatial)')
args = parser.parse_args()

bidir = args.bidir

# Initialize wandb
if args.wandb:
    ########################################################### CHANGE THIS FOR MSE+CX and MSE
    run = wandb.init(project='voxelmorph',group= 'multimodal_96crop',
                     name= args.loss_feat+'_p8_sim01_newl2l2_lambda='+str(args.weight),
    config={'encoder filters':args.enc, 'decoder filters':args.dec, 'int-steps':args.int_steps,
            'int-downsize':args.int_downsize, 'type of loss':args.loss_rgb+'&'+args.loss_feat,
            'number-epochs:':args.epochs, 'steps per epoch':args.steps_per_epoch,'lr':args.lr,
            'similarity:':args.similarity,'weight-sp':args.weight_sp, 'layer':2, 'patch_size':(8,8,8),
            'pretrained network':'adni', 'stride':8, 'data_train':100,
            })


# load and prepare training data
# dict with keys:image,label;values:file_names
dataset = load_data_multimodal(data_dir=args.data_dir, csv_file = '/home/franavarro25/TUM/Master_thesis/ANTs_cpp/non_valid_uids.csv') 

# dataloader--> dict with keys: image,labels,image_metadata,label_metadata; values: real 3d volumes 
training_loader, val_loader = data_loader_multimodal(dataset = dataset, 
                                      batch_size=args.batch_size) 

# extract shape from sampled input [2(images),1,160,192,224]]
inshape = next(iter(training_loader))['image1'].shape[-3:]
print('inshape', inshape)

# device handling
gpus = args.gpu.split(',')
nb_gpus = len(gpus)
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
assert np.mod(args.batch_size, nb_gpus) == 0, \
    'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_gpus)

# enabling cudnn determinism appears to speed up training by a lot
torch.backends.cudnn.deterministic = not args.cudnn_nondet ### where does this affect???


# prepare model and saving path
# for debugging --> model_n=args.model_output
# losses experiments --> model_n = args.loss_rgb+'_'+args.loss_feat+'/our_loss_patches/'+'lambda_'+str(args.weight)
# long training -->model_n=args.image_loss+'/'+args.model_output
torch.cuda.reset_max_memory_allocated()
max_mem = torch.cuda.max_memory_allocated() / 2**20 
print('memoria del voxelmorph antes', max_mem)
model, model_dir = vxm_model(model_n = 'multimodal_crop96_p8_sim01_newl2l2_'+args.loss_feat+'/'+'lambda_'+str(args.weight), 
                             load_model=args.load_model, device=device, enc=args.enc, 
                             dec=args.dec, in_shape=inshape, bidir=args.bidir,
                             int_steps=args.int_steps, int_downsize=args.int_downsize)
max_mem = torch.cuda.max_memory_allocated() / 2**20 
print('memoria del voxelmorph', max_mem)

if nb_gpus > 1:
    # use multiple GPUs via DataParallel
    model = torch.nn.DataParallel(model)
    model.save = model.module.save

# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# prepare image loss

losses, weights = loss_functions(image_loss=args.loss_feat, bidir=args.bidir, weight=args.weight, weight_sp=args.weight_sp,
                                int_downsize=args.int_downsize, input_shape = inshape, similarity=args.similarity)
'''
losses, weights = two_loss_functions(input_shape = inshape, image_loss = args.loss_rgb, 
                                    loss_vgg = args.loss_feat, similarity=args.similarity, 
                                    weight_sp=args.weight_sp, 
                                    bidir=args.bidir, weight=args.weight, 
                                    int_downsize=args.int_downsize, device=device)
'''
print('numero de losses', len(losses), losses)
transform_model = vxm.layers.SpatialTransformer(inshape, mode='nearest').to(device)
s=0

brainOneHotd = BrainOneHotd(keys='seg')

compute_jackdet = SDlogDetJac()
for epoch in range(args.initial_epoch, args.epochs):
    s+=1
    model.train()
    # save model checkpoint
    if epoch % 20 == 0 or (epoch+1)==args.epochs:
        if epoch % 20 ==0: 
            model.save(os.path.join(model_dir, '%04d.pt' % epoch))
        model.eval()
        dice_score = []
        hausdorff_distance = []
        SDlog_Jac = []
        non_pos_jacdet = []
        epoch_val_loss = []
        epoch_val_total_loss = []
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                #if i == 1:
                    #print('imagen1', data['image1_meta_dict']['filename_or_obj'])
                    #print('imagen2', data['image2_meta_dict']['filename_or_obj'])
                    #print('seg_val', data['seg_meta_dict']['filename_or_obj'])
                    #data is a dict--> keys:('image',label); values:(2*batch_sizex1ximage_size,2*batch_sizex1xmask_size)   
                    moved_mask, fixed_mask, target, y_source, flow = run_epoch_multimodal(data=data, model=model, 
                                                                        transform_model=transform_model,
                                                                        device=device, task='val',brainOneHotd=brainOneHotd)
                    sdlog_jacdet_step, non_pos_jacdet_step = compute_jackdet(flow.detach().cpu())
                    #just to check
                    #y_source_mask = data['seg'][1].unsqueeze(0).float().to(device)
                    #y_source_mask = {'seg': y_source_mask}
                    #moved_mask = brainOneHotd(y_source_mask)['seg'].unsqueeze(0)
                    # calculate the dice
                    # monai.metrics.meandice.compute_dice(y_pred=moved_mask, y=fixed_mask)(20x36)
                    dice_score_labels = monai.metrics.meandice.compute_dice(y_pred=moved_mask, y=fixed_mask, include_background=False)
                    #dice_score_mask = torch.isnan(dice_score_labels)
                    #dice_score_labels[dice_score_mask] = 0.0
                    #dice_score_mask = ~torch.isnan(dice_score_labels)
                    #dice_score_labels = dice_score_labels[dice_score_mask]
                    dice_step = torch.mean(dice_score_labels)
                    hausdorff_distance_labels =  monai.metrics.hausdorff_distance.compute_hausdorff_distance(moved_mask, fixed_mask)
                    #hausdorff_distance_mask = ~torch.isnan(hausdorff_distance_labels)
                    #hausdorff_distance_labels = hausdorff_distance_labels[hausdorff_distance_mask]
                    hausdorff_distance_step = torch.mean(hausdorff_distance_labels)
                    
                    dice_score.append(dice_step)
                    hausdorff_distance.append(hausdorff_distance_step)
                    SDlog_Jac.append(sdlog_jacdet_step)
                    non_pos_jacdet.append(non_pos_jacdet_step)


                    loss_list = []
                    # calculate validation loss
                    # wsp = 2 * torch.log10(torch.tensor(500/(epoch+1))) / torch.log10(torch.tensor(500))
                    loss0 = losses[0](y_source, target) * weights[0]
                    loss_list.append(loss0.item())
                    loss1 = losses[1](flow, torch.zeros((flow.shape))) * weights[1]
                    loss_list.append(loss1.item())
                    loss = loss0 + loss1

                    epoch_val_loss.append(loss_list)
                    epoch_val_total_loss.append(loss.item())
        
        epoch_val_loss = np.mean(epoch_val_loss, axis=0)
        epoch_val_total_loss = sum(epoch_val_total_loss)/len(epoch_val_total_loss)
        dice_score = sum(dice_score)/len(dice_score)
        hausdorff_distance = sum(hausdorff_distance)/len(hausdorff_distance)
        SDlog_Jac = sum(SDlog_Jac)/len(SDlog_Jac)
        non_pos_jacdet = sum(non_pos_jacdet)/len(non_pos_jacdet)
        if  args.wandb:
    	    wandb.log({'dice_score(val)':dice_score.item(),
                       'hausdorff_distance ':hausdorff_distance.item(),
                       'SDlog_Jac':SDlog_Jac.item(),
                       'non_pos_jacdet':non_pos_jacdet.item(),
                       'val_total_loss':epoch_val_total_loss,
                       #'rgb_loss':epoch_val_loss[0],
                       'feat_loss':epoch_val_loss[0],
                       'reg_loss':epoch_val_loss[1]}, step=s)
        print('dice_score', dice_score.item(), 'hausdorff_distance ',hausdorff_distance.item())
        print('SDlog_Jac', SDlog_Jac.item(),'non_pos_jacdet', non_pos_jacdet.item())
        print('epoch_val_losses', epoch_val_loss[0], epoch_val_loss[1])
        print('epcoh_val_total_loss', epoch_val_total_loss)
        torch.cuda.empty_cache()
        del moved_mask, fixed_mask, target, y_source, flow, loss0, loss1, loss
        model.train()

    epoch_loss = []
    epoch_total_loss = []
    epoch_step_time = []
    
    for i, data in enumerate(training_loader):
        #if i == 1:
            #print('imagen1', data['image1_meta_dict']['filename_or_obj'])
            #print('imagen2', data['image2_meta_dict']['filename_or_obj'])
            #print('seg_train', data['seg_meta_dict']['filename_or_obj'])
            
            '''sitk.WriteImage(sitk.GetImageFromArray(data['image2'][1].unsqueeze(0).float()), '/home/franavarro25/Desktop/loaded_images/'+str(i)+'_source.nii.gz')
            sitk.WriteImage(sitk.GetImageFromArray(data['image1'][0].unsqueeze(0).float()), '/home/franavarro25/Desktop/loaded_images/'+str(i)+'_target.nii.gz')
            sitk.WriteImage(sitk.GetImageFromArray(data['seg'][1].unsqueeze(0).float()), '/home/franavarro25/Desktop/loaded_images/'+str(i)+'_source_mask.nii.gz')
            sitk.WriteImage(sitk.GetImageFromArray(data['seg'][0].unsqueeze(0).float()), '/home/franavarro25/Desktop/loaded_images/'+str(i)+'_target_mask.nii.gz')'''
            
            step_start_time = time.time()
            
            source, target, y_source, flow = run_epoch_multimodal(
                data=data, model=model, transform_model=transform_model,device=device, task='train', brainOneHotd=brainOneHotd)
            
            #print('target shape to divide', type(target))
            #print('y_source shape to divide', type(y_source))
            #print('flow shape to divide', type(flow))
            
            loss_list = []
            # calculate total loss
            loss0 = losses[0](y_source, target) * weights[0]
            loss_list.append(loss0.item())
            loss1 = losses[1](flow, torch.zeros((flow.shape))) * weights[1]
            loss_list.append(loss1.item())
            loss = loss0 + loss1
            
            epoch_loss.append(loss_list)
            epoch_total_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del  source, target, y_source, flow

            # get compute time
            epoch_step_time.append(time.time() - step_start_time)

    # print epoch info
    epoch_info = 'Epoch %d/%d' % (epoch , args.epochs)
    time_info = '%.4f sec/step' % np.mean(epoch_step_time)
    losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
    loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
    #print(' - '.join((epoch_info, time_info, loss_info)), flush=True)

    if  args.wandb:
    	wandb.log({'epoch':epoch, 'time(sec/step)':np.mean(epoch_step_time), 'loss':(np.mean(epoch_total_loss)).item(),
                    #'loss_rgb':np.mean(epoch_loss,axis=0)[0],
                    'loss_feat':np.mean(epoch_loss,axis=0)[0],
                    'loss_reg':np.mean(epoch_loss,axis=0)[1]}, step=s)
    print(' - '.join((epoch_info, time_info, loss_info)), flush=True)

if args.wandb:
    run.finish()

# final model save
model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))