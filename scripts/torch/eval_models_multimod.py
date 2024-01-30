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

from utils import *
from losses import contextual_bilateral_loss3D_local
import matplotlib.pyplot as plt
import numpy as np
import wandb

# import my utils
from data_utils import load_data_multimodal, data_loader_multimodal, BrainOneHotd
from vxm_network import vxm_model
from losses import loss_functions, two_loss_functions
from utils import run_epoch_multimodal, divide_volume
from metrics import SDlogDetJac

loss_fn = 'mind' # mind, mi, cobi_patches_local, cobi_contra_patches_local
wandb_ = False

if wandb_:
    run = wandb.init(project='voxelmorph',group= 'multimodal_dsc',
                     name= loss_fn)


# load and prepare training data
# dict with keys:image,label;values:file_names
dataset = load_data_multimodal(data_dir='/home/franavarro25/TUM/Master_thesis/dataset/ADNI_remote/ADNI_T1_t1space/', 
                               csv_file = '/home/franavarro25/TUM/Master_thesis/ANTs_cpp/non_valid_uids.csv') 

# dataloader--> dict with keys: image,labels,image_metadata,label_metadata; values: real 3d volumes 
training_loader, val_loader = data_loader_multimodal(dataset = dataset, 
                                                     batch_size=1) 

# extract shape from sampled input [2(images),1,160,192,224]]                                                     
inshape = next(iter(training_loader))['image1'].shape[-3:]

# loss we are going to use
losses, weights = loss_functions(image_loss=loss_fn, weight=1, weight_sp=0.1,bidir=False,
                                int_downsize=2, input_shape = inshape, similarity='cosine')


device = 'cuda'
transform_model = vxm.layers.SpatialTransformer(inshape, mode='nearest').to(device)

brainOneHotd = BrainOneHotd(keys='seg')

s=0

compute_jackdet = SDlogDetJac()

# folders with the models to load
models_folder = '/home/franavarro25/TUM/Master_thesis/voxelmorph_monai/models/multimodal'+loss_fn+'/lambda_1/'
model_names = sorted(os.listdir(models_folder))
print('model_names', model_names)

for model_name in model_names:
    # load the model
    print('model name:', model_name)
    model_path = models_folder+model_name
    model = vxm.networks.VxmDense.load(model_path, device).cuda()
    model.eval()
    
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    dice_score = []
    hausdorff_distance = []
    SDlog_Jac = []
    non_pos_jacdet = []
    epoch_val_loss = []
    epoch_val_total_loss = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            #data is a dict--> keys:('image',label); values:(2*batch_sizex1ximage_size,2*batch_sizex1xmask_size)   
            moved_mask, fixed_mask, target, y_source, flow = run_epoch_multimodal(data=data, model=model, 
                                                                    transform_model=transform_model,
                                                                    device=device, task='val',brainOneHotd=brainOneHotd)
            
            sdlog_jacdet_step, non_pos_jacdet_step = compute_jackdet(flow.detach().cpu())

            # calculate the dice
            # monai.metrics.meandice.compute_dice(y_pred=moved_mask, y=fixed_mask)(20x36)
            dice_score_labels = monai.metrics.meandice.compute_dice(y_pred=moved_mask, y=fixed_mask)
            dice_step = torch.mean(dice_score_labels)
            hausdorff_distance_labels =  monai.metrics.hausdorff_distance.compute_hausdorff_distance(moved_mask, fixed_mask)
            hausdorff_distance_step = torch.mean(hausdorff_distance_labels)
            
            dice_score.append(dice_step)
            hausdorff_distance.append(hausdorff_distance_step)
            SDlog_Jac.append(sdlog_jacdet_step)
            non_pos_jacdet.append(non_pos_jacdet_step)


            loss_list = []
            # calculate validation loss
            
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
    if wandb_:    
        wandb.log({'dice_score(val)':dice_score.item(),
               'hausdorff_distance ':hausdorff_distance.item(),
               'SDlog_Jac':SDlog_Jac.item(),
               'non_pos_jacdet':non_pos_jacdet.item(),
               'val_total_loss':epoch_val_total_loss,
               #'rgb_loss':epoch_val_loss[0],
               'feat_loss':epoch_val_loss[0],
               'reg_loss':epoch_val_loss[1]}, step=s)
    s += 1
    print('dice_score', dice_score.item(), 'hausdorff_distance ',hausdorff_distance.item())
    print('SDlog_Jac', SDlog_Jac.item(),'non_pos_jacdet', non_pos_jacdet.item())
    print('epoch_val_losses', epoch_val_loss[0], epoch_val_loss[1])
    print('epcoh_val_total_loss', epoch_val_total_loss)
    torch.cuda.empty_cache()
    del moved_mask, fixed_mask, target, y_source, flow, loss0, loss1, loss


run.finish()