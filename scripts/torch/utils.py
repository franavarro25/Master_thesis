import monai
from data_utils import BrainOneHotd

'''
Script to run one epoch of the different experiments
run_epoch: one epoch for OASIS dataset
run_epoch_multi-modal_t11loss: one multi-modal epoch on ADNI dataset with mono-modal loss
run_epoch_multi-modal: one multi-modal epoch on ADNI dataset
'''

def run_epoch(data, model, transform_model, device, task):
    source = data['image'][0].unsqueeze(0).float().to(device)
    target = data['image'][1].unsqueeze(0).float().to(device)
    source_mask = data['label'][0].unsqueeze(0).float().to(device) # comment .to(device)
    target_mask = data['label'][1].unsqueeze(0).float().to(device) # if there is no gpu space
    
    # run inputs through the model to produce a warped image and flow field
    y_source, flow = model(source, target, registration=True)
                            # y_source: tensor(20x1x160x192x224); flow: tensor(20x3x80x96x112) 
                            # con registration 20x3x160x192x224
    if task=='train': return source, target, y_source, flow
        #del source_mask, target_mask
    elif task=='val': 
        #print(y_source.shape)
        # ahora la mask (20x1x160x192x224)
        y_source_mask = transform_model(source_mask, flow)

        # binarize mask (20x36x160x192x224)
        moved_mask = monai.networks.utils.one_hot(y_source_mask, num_classes=36)
        fixed_mask = monai.networks.utils.one_hot(target_mask, num_classes=36)
        
        return moved_mask, fixed_mask, target, y_source, flow

def run_epoch_multimodal_t11loss(data, model, transform_model, transformer, device, task, brainOneHotd):
    # this numbers becuse target/fixed is T1 and moving/source is T2
    source_t2 = data['image2'][1].unsqueeze(0).float().to(device)
    target_t1 = data['image1'][0].unsqueeze(0).float().to(device)
    source_t1 = data['image1'][1].unsqueeze(0).float().to(device)
    target_t2 = data['image2'][0].unsqueeze(0).float().to(device)
    source_mask = data['seg'][1].unsqueeze(0).float().to(device)
    target_mask = data['seg'][0].unsqueeze(0).float().to(device)
    
    # run inputs through the model to produce a warped image and flow field
    y_source_t2, flow = model(source_t2, target_t1, registration=True)
                            # y_source: tensor(20x1x160x192x224); flow: tensor(20x3x80x96x112) 
                            # con registration 20x3x160x192x224
    y_source_t1 = transformer(source_t1, flow)
    if task=='train': return source_t1, target_t1, y_source_t1, flow 
        #del source_mask, target_mask
    elif task=='val': 
        #print(y_source.shape)
        # ahora la mask (20x1x160x192x224)
        y_source_mask = transform_model(source_mask, flow)
        #print('ysourcmask', y_source_mask.shape)
        #print(torch.unique(y_source_mask), torch.unique(target_mask))
        y_source_mask = {'seg': y_source_mask}
        target_mask = {'seg': target_mask}
        
        # binarize mask (20x36x160x192x224)
        moved_mask = brainOneHotd(y_source_mask)['seg'].unsqueeze(0)
        fixed_mask = brainOneHotd(target_mask)['seg'].unsqueeze(0)
        #print(moved_mask.shape, fixed_mask.shape, target.shape, y_source.shape, flow.shape)
        
        return moved_mask, fixed_mask, target_t1, y_source_t1, flow


def run_epoch_multimodal(data, model, transform_model, device, task, brainOneHotd):
    # this numbers becuse target/fixed is T1 and moving/source is T2
    source = data['image2'][1].float().unsqueeze(0).to(device)
    target = data['image1'][0].float().unsqueeze(0).to(device)
    source_mask = data['seg'][1].float().unsqueeze(0).to(device)
    target_mask = data['seg'][0].float().unsqueeze(0).to(device)
    
    # run inputs through the model to produce a warped image and flow field
    y_source, flow = model(source, target, registration=True)
                            # y_source: tensor(20x1x160x192x224); flow: tensor(20x3x80x96x112) 
                            # con registration 20x3x160x192x224
    if task=='train': return source, target, y_source, flow 
        #del source_mask, target_mask
    elif task=='val': 
        #print(y_source.shape)
        # ahora la mask (20x1x160x192x224)
        #print('flowtype', flow.dtype)
        y_source_mask = transform_model(source_mask, flow)
        #print('ysourcmask', y_source_mask.shape)
        #print(torch.unique(y_source_mask), torch.unique(target_mask))
        y_source_mask = {'seg': y_source_mask}
        target_mask = {'seg': target_mask}
        
        # binarize mask (20x36x160x192x224)
        moved_mask = brainOneHotd(y_source_mask)['seg'].unsqueeze(0)
        fixed_mask = brainOneHotd(target_mask)['seg'].unsqueeze(0)
        #print(moved_mask.shape, fixed_mask.shape, target.shape, y_source.shape, flow.shape)
        
        return moved_mask, fixed_mask, target, y_source, flow




import torch
import torch.nn.functional as F
import numpy as np

'''
Some functions used during the project, and mainly used in the main code 
to compute the loss functions
'''
def compute_cx(dist_tilde, band_width):
    w = torch.exp((1 - dist_tilde) / band_width)  # Eq(3)
    cx = w / torch.sum(w, dim=2, keepdim=True)  # Eq(4)
    return cx

def compute_relative_distance(dist_raw):
    dist_min, _ = torch.min(dist_raw, dim=2, keepdim=True) #original dim=2
    dist_tilde = dist_raw / (dist_min + 1e-5)
    return dist_tilde # eq(2)



def compute_cosine_similarity_3d(x, y):
    N, C, *_ = x.size()
    x = x.reshape(N, C, -1)  # (N, C, H*W*D)
    y = y.reshape(N, C, -1)  # (N, C, H*W*D)

    # consine similarity
    cosine_sim = torch.bmm(x.transpose(1, 2),y)  # (N, H*W*D, H*W*D)
    
    # move to [0,1] range
    cosine_sim = (cosine_sim+1)/2

    return cosine_sim


def compute_l1_distance_3d(x: torch.Tensor, y: torch.Tensor):
    N, C, H, W, D = x.size()
    x_vec = x.view(N, C, -1)
    y_vec = y.view(N, C, -1)
    
    dist = x_vec.unsqueeze(2) - y_vec.unsqueeze(3)
    dist = dist.abs().sum(dim=1)
    dist = dist.transpose(1, 2).reshape(N, H*W*D, H*W*D)
    dist = dist.clamp(min=0.)

    return dist

def compute_triangle(dist_raw):
    similarity = 1 - dist_raw

    theta = torch.arccos(similarity) #+ np.radians(10)
    TS = torch.sin(theta) / 2
    return TS, theta

def compute_sector(x, y):
    #Magn_dif = torch.abs(torch.linalg.norm(x, p=2, dim=1, keepdim=True) - torch.linalg.norm(y, p=2, dim=1, keepdim=True).T)
    magn_dif = torch.abs(compute_l1_distance_3d(x,y))
    eucl_dis = compute_l2_distance_3d(x, y)
    return (eucl_dis+magn_dif)**2

def compute_ts_ss(x, y, dist_raw):
    TS, theta = compute_triangle(dist_raw)
    sector = compute_sector(x, y)
    #print(dist_raw[0,0,255], dist_raw[0,43,43])
    if torch.isnan(sector).any(): print('sector', torch.argwhere(torch.isnan(sector)))
    if torch.isnan(theta).any():print('theta', torch.argwhere(torch.isnan(theta)))
    if torch.isnan(TS).any():print('TS', torch.argwhere(torch.isnan(TS)))
    return TS*sector*theta/2

def compute_l2_distance_3d(x, y):
    N, C, H, W, D = x.size()
    x_vec = x.view(N, C, -1)
    y_vec = y.view(N, C, -1)
    x_s = torch.sum(x_vec ** 2, dim=1, keepdim=True)
    y_s = torch.sum(y_vec ** 2, dim=1, keepdim=True)
    A = y_vec.transpose(1, 2) @ x_vec
    dist = y_s - 2 * A + x_s.transpose(1, 2)
    dist = dist.transpose(1, 2).reshape(N, H*W*D, H*W*D)
    dist = dist.clamp(min=0.)

    return dist



def compute_meshgrid_3d(shape):
    N, C, H, W, D = shape
    rows = torch.arange(0, H, dtype=torch.float32) / (H + 1)
    cols = torch.arange(0, W, dtype=torch.float32) / (W + 1)
    depths = torch.arange(0, D, dtype=torch.float32) / (D + 1)

    feature_grid = torch.meshgrid(rows, cols, depths)
    feature_grid = torch.stack(feature_grid).unsqueeze(0)
    feature_grid = torch.cat([feature_grid for _ in range(N)], dim=0)

    return feature_grid

def divide_volume(volume, patch_size):
    volume_shape = volume.shape # BxCxHxWxD
    
    patch_stride = patch_size #[patch_size[i] // 2 for i in range(3)] or patch_size
    patches = volume.unfold(2, patch_size[0], patch_stride[0]).unfold(3, patch_size[1], patch_stride[1]).unfold(4, patch_size[2], patch_stride[2])
    patches = patches.contiguous().view(volume_shape[0], volume_shape[1], -1, patch_size[0], patch_size[1], patch_size[2]).permute(2,0,1,3,4,5)
    
    return patches

def divide_volume_attention(volume, patch_size):
    volume_shape = volume.shape # BxCxHxWxD
    
    patch_stride = patch_size #[patch_size[i] // 2 for i in range(3)] or patch_size
    patches = volume.unfold(2, patch_size[0], patch_stride[0]).unfold(3, patch_size[1], patch_stride[1]).unfold(4, patch_size[2], patch_stride[2])
    patches = patches.contiguous().view(volume_shape[0], volume_shape[1], -1, patch_size[0] * patch_size[1] * patch_size[2]).permute(2,0,1,3)
    
    return patches


def random_select_element(volume1, volume2, n_elements):
    B,C,H,W,D = volume1.shape
    reshaped_tensor1 = volume1.view(B,C,-1).permute(2,0,1) # H*w*DxBxC 
    reshaped_tensor2 = volume2.view(B,C,-1).permute(2,0,1) # H*w*DxBxC 
    total_elements = reshaped_tensor1.shape[0]
    random_indices = np.random.choice(total_elements, size=n_elements, replace=False)
    selected_elements1 = reshaped_tensor1[random_indices] # n_elementsxBxC 
    selected_elements2 = reshaped_tensor2[random_indices] # n_elementsxBxC 
    selected_elements1 = selected_elements1.permute(1,2,0) # BxCxn_elements
    selected_elements2 = selected_elements2.permute(1,2,0) # BxCxn_elements
    return selected_elements1, selected_elements2

def positives_negatives_(similarity_matrix):
    # Get the dimensions of the similarity matrix   
    batch_size, num_rows, num_cols = similarity_matrix.shape  
    
    # Create a diagonal matrix(mask) with the same shape as the input tensor
    diag_mask = torch.eye(num_rows, num_cols).view(batch_size,num_rows,num_cols).bool()
    # take the diagonal values as the positives
    positives = similarity_matrix[diag_mask]
    
    # Create a mask to identify the non-diagonal elements
    non_diag_mask = ~diag_mask
    # create the negatives
    negatives = []
    for i in range(num_cols):
      negatives.append(similarity_matrix[0,:,i][non_diag_mask[0,:,i]])
    
    negatives = torch.stack(negatives, dim=0).T
    
    return positives.unsqueeze(0), negatives.unsqueeze(0)

def positives_negatives(similarity_matrix):
    # Get the dimensions of the similarity matrix
    batch_size, num_rows, num_cols = similarity_matrix.shape  
    
    # Create a diagonal mask and use it to get positives
    diag_mask = torch.eye(num_rows, num_cols, device=similarity_matrix.device).expand(batch_size,-1,-1).bool()
    positives = similarity_matrix[diag_mask].view(batch_size, -1)
    
    # Create a mask to identify the non-diagonal elements
    non_diag_mask = ~diag_mask
    
    # Expand the non_diag_mask to match the batch_size
    non_diag_mask = non_diag_mask.expand(batch_size, -1, -1)
    
    # Create negatives using the mask
    negatives = similarity_matrix.permute(0,2,1)[non_diag_mask].view(batch_size, num_rows, num_cols-1).permute(0,2,1)
    
    return positives, negatives

def positives_negatives2(similarity_matrix):
    # Get the dimensions of the similarity matrix
    batch_size, num_rows, num_cols = similarity_matrix.shape

    # Create a diagonal mask and use it to get positives
    diag_mask = torch.eye(num_rows, num_cols, device=similarity_matrix.device).expand(batch_size,-1,-1).bool()
    positives = similarity_matrix[diag_mask].view(batch_size, -1)

    # Create a mask to identify the non-diagonal elements
    non_diag_mask = ~diag_mask

    # Expand the non_diag_mask to match the batch_size
    non_diag_mask = non_diag_mask.expand(batch_size, -1, -1)

    # Create negatives using the mask
    negatives = similarity_matrix[non_diag_mask].view(batch_size, num_rows, num_cols-1)

    return positives, negatives