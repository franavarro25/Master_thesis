import os
# import voxelmorph with pytorch backend from source
os.environ['NEURITE_BACKEND'] = 'pytorch'  ###############333
os.environ['VXM_BACKEND'] = 'pytorch'      ############
import sys 
sys.path.append('/home/franavarro25/TUM/Master_thesis/voxelmorph_monai')
import voxelmorph as vxm
sys.path.append('/home/franavarro25/TUM/Master_thesis')
sys.path.append('/home/franavarro25/TUM/Master_thesis/MedicalNet')
import MedicalNet as MedNet
from MedicalNet.load_pretrained import load_function
import monai
from utils import divide_volume, compute_meshgrid_3d

'''
In this script we can find all the loss functions that we have used and tried in this project: 
    - NCC: from voxelmorph 
    - MSE: from voxelmorph
    - MI: from monai 
    - MIND: from Bailiang (advisor)
    - NGF: from Bailiang (advisor)
    - Contextual loss: original loss function calculated for patches
    - Contextual loss + spatial info: all versions of the loss function 
    - Contrastive loss: used to pretrain the resnet 
'''
def loss_functions(image_loss, bidir, weight, int_downsize, input_shape, similarity, weight_sp):
    '''
    - With this function we choose: similarity loss and regularization loss
    '''
    if image_loss.lower() == 'ncc':
        image_loss_func = vxm.losses.NCC().loss
    elif image_loss.lower() == 'mse':
        image_loss_func = vxm.losses.MSE().loss
    elif image_loss.lower() == 'mi':
        image_loss_func = monai.losses.image_dissimilarity.GlobalMutualInformationLoss(num_bins=32)
    elif image_loss.lower() == 'mind':
        image_loss_func = MINDSSCLoss(radius=2,dilation=2)
    elif image_loss == 'ngf':
        image_loss_func = NormalizedGradientField3d(mm_spacing=1)
    elif image_loss.lower() == 'cx_patches':
        image_loss_func = ContextualLoss3D_patches(vgg_vol_shape=input_shape,loss_type = similarity)
    elif image_loss.lower() == 'cx_patches_local':
        image_loss_func = ContextualLoss3D_patches_local(vgg_vol_shape=input_shape,loss_type = similarity)
    elif image_loss.lower() == 'cobi_patches_local_mm':
        image_loss_func = ContextualBilateralLoss3D_local_mm(vgg_vol_shape=input_shape,weight_sp=weight_sp,loss_type=similarity)
    elif image_loss.lower() == 'cobi_patches_local':
        image_loss_func = ContextualBilateralLoss3D_local(vgg_vol_shape=input_shape,weight_sp=weight_sp,loss_type=similarity)
    elif image_loss.lower() == 'cobi_patches_local_multilayer':
        image_loss_func = ContextualBilateralLoss3D_local_multilayer(vgg_vol_shape=input_shape,weight_sp=weight_sp,loss_type=similarity)
    elif image_loss.lower() == 'cobi_patches':
        image_loss_func = ContextualBilateralLoss3D(vgg_vol_shape=input_shape,weight_sp=weight_sp,loss_type=similarity)
    else:
        raise ValueError('Image loss "%s" not found' % args.image_loss)

    # need two image loss functions if bidirectional
    if bidir:
        losses = [image_loss_func, image_loss_func]
        weights = [0.5, 0.5]
    else:
        losses = [image_loss_func]
        weights = [1]

    # prepare deformation loss
    losses += [vxm.losses.Grad('l2', loss_mult=int_downsize).loss]
    weights += [weight]

    return losses, weights

def two_loss_functions(input_shape, image_loss, loss_vgg, similarity, weight_sp, bidir, weight, int_downsize, device):
    # function to calculate Intensity similarity + feature similarity + regularization 
    if image_loss.lower() == 'ncc':
        image_loss_func = vxm.losses.NCC().loss
    elif image_loss.lower() == 'mse':
        image_loss_func = vxm.losses.MSE().loss
    elif image_loss.lower() == 'mi':
        image_loss_func = monai.losses.image_dissimilarity.GlobalMutualInformationLoss()
    elif image_loss.lower() == 'mind':
        image_loss_func = MINDSSCLoss(radius=2,dilation=2)
    elif image_loss == 'ngf':
        image_loss_func = NormalizedGradientField3d(mm_spacing=1)
    elif loss_vgg.lower() == 'cx_patches':
        feature_crit = ContextualLoss3D_patches(vgg_vol_shape=input_shape,loss_type = similarity)
    elif loss_vgg.lower() == 'cx_patches_local':
        feature_crit = ContextualLoss3D_patches_local(vgg_vol_shape=input_shape,loss_type = similarity)
    elif loss_vgg.lower() == 'cobi':
        feature_crit = ContextualBilateralLoss3D(vgg_vol_shape=input_shape,weight_sp=weight_sp,loss_type=similarity)
    else:
        raise ValueError('Image loss should be "l1", "l2", cx or cobi, but found "%s"' % args.image_loss)
    image_loss_func = [image_loss_func]+[feature_crit]

    # need two image loss functions if bidirectional
    if bidir:
        # need to change this
        losses = [image_loss_func, image_loss_func]
        weights = [0.5, 0.5]
    else:
        losses = image_loss_func
        weights = [1,1]

    # prepare deformation loss
    losses += [vxm.losses.Grad('l2', loss_mult=int_downsize).loss]
    weights += [weight]

    return losses, weights


##########################################################################
# Functions and code from Bailiang (advisor) to compute MIND loss

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ConstantPad3d, ReplicationPad3d

from builder_bailiang import LOSSES


def pdist_squared(x: torch.Tensor) -> torch.Tensor:
    """Compute the pairwise squared euclidean distance of input coordinates.
    Args:
        x: input coordinates, input shape should be (1, dim, #input points)
    Returns:
        dist: pairwise distance matrix, (#input points, #input points)
    """
    xx = (x**2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, np.inf)
    return dist


@LOSSES.register_module('mind')
class MINDSSCLoss(nn.Module):
    """
    Modality-Independent Neighbourhood Descriptor Dissimilarity Loss for Image Registration
    References: https://link.springer.com/chapter/10.1007/978-3-642-40811-3_24
    Args:
        radius (int): radius of self-similarity context.
        dilation (int): the dilation of neighbourhood patches.
        penalty (str): the penalty mode of mind dissimilarity loss.
    """
    def __init__(
        self,
        radius: int = 2,
        dilation: int = 2,
        penalty: str = 'l2',
    ) -> None:
        super().__init__()
        self.kernel_size = radius * 2 + 1
        self.dilation = dilation
        self.radius = radius
        self.penalty = penalty
        self.mshift1, self.mshift2, self.rpad1, self.rpad2 = self.build_kernels(
        )

    def build_kernels(self):
        # define start and end locations for self-similarity pattern
        six_neighbourhood = torch.Tensor([[0, 1, 1], [1, 1, 0], [1, 0, 1],
                                          [1, 1, 2], [2, 1, 1], [1, 2,
                                                                 1]]).long()

        # squared distances
        dist = pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

        # define comparison mask, square distance equals 2
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        # build kernel
        # self-similarity context: 12 elements
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6,
                                                           1).view(-1,
                                                                   3)[mask, :]
        mshift1 = torch.zeros(12, 1, 3, 3, 3)
        mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 +
                         idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
        mshift1.requires_grad = False

        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1,
                                                           1).view(-1,
                                                                   3)[mask, :]
        mshift2 = torch.zeros(12, 1, 3, 3, 3)
        mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 +
                         idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
        mshift2.requires_grad = False

        # maintain the output size
        rpad1 = ReplicationPad3d(self.dilation)
        rpad2 = ReplicationPad3d(self.radius)
        return mshift1, mshift2, rpad1, rpad2

    def mind(self, img: torch.Tensor) -> torch.Tensor:
        mshift1 = self.mshift1.to(img)
        mshift2 = self.mshift2.to(img)
        # compute patch-ssd
        ssd = F.avg_pool3d(self.rpad2(
            (F.conv3d(self.rpad1(img), mshift1, dilation=self.dilation) -
             F.conv3d(self.rpad1(img), mshift2, dilation=self.dilation))**2),
                           self.kernel_size,
                           stride=1)

        # MIND equation
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        mind_var = torch.clamp(mind_var,
                               mind_var.mean() * 0.001,
                               mind_var.mean() * 1000)
        mind = torch.div(mind, mind_var)
        mind = torch.exp(-mind)

        # permute to have same ordering as C++ code
        mind = mind[:,
                    torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(
                    ), :, :, :]

        return mind

    def forward(self, source: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """Compute the MIND-SSC loss.
        Args:
            source: source image, tensor of shape [BNHWD].
            target: target image, tensor fo shape [BNHWD].
        """
        assert source.shape == target.shape, 'input and target must have the same shape.'
        if self.penalty == 'l1':
            mind_loss = torch.abs(self.mind(source) - self.mind(target))
        elif self.penalty == 'l2':
            mind_loss = torch.square(self.mind(source) - self.mind(target))
        else:
            raise ValueError(
                f'Unsupported penalty mode: {self.penalty}, available modes are l1 and l2.'
            )

        return torch.mean(mind_loss)  # the batch and channel average

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += (f'(radius={self.radius},'
                     f'dilation={self.dilation},'
                     f'penalty=\'{self.penalty}\')')
        return repr_str

##########################################################################################

 
########################################################################################################
# Functions and code from Bailiang (advisor) to compute NGF loss

import warnings
from typing import Callable, List, Optional, Sequence, Union, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from kernels_bailiang import gauss_kernel_1d, gauss_kernel_2d, gauss_kernel_3d
from kernels_bailiang import gradient_kernel_1d, gradient_kernel_2d, gradient_kernel_3d
from kernels_bailiang import spatial_filter_nd
from torch.nn.parameter import Parameter
from monai.utils.enums import LossReduction

def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return [x, x]


def _grad_param(ndim, method, axis):
    if ndim == 1:
        kernel = gradient_kernel_1d(method)
    elif ndim == 2:
        kernel = gradient_kernel_2d(method, axis)
    elif ndim == 3:
        kernel = gradient_kernel_3d(method, axis)
    else:
        raise NotImplementedError

    kernel = kernel.reshape(1, 1, *kernel.shape)
    return Parameter(torch.Tensor(kernel).float())


def _gauss_param(ndim, sigma, truncate):
    if ndim == 1:
        kernel = gauss_kernel_1d(sigma, truncate)
    elif ndim == 2:
        kernel = gauss_kernel_2d(sigma, truncate)
    elif ndim == 3:
        kernel = gauss_kernel_3d(sigma, truncate)
    else:
        raise NotImplementedError

    kernel = kernel.reshape(1, 1, *kernel.shape)
    return Parameter(torch.Tensor(kernel).float())


class NormalizedGradientField3d(_Loss):
    """
    Compute the normalized gradient fields defined in:
    Haber, Eldad, and Jan Modersitzki. "Intensity gradient based registration and fusion of multi-modal images."
    In International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 726-733. Springer,
    Berlin, Heidelberg, 2006.
    Häger, Stephanie, et al. "Variable Fraunhofer MEVIS RegLib Comprehensively Applied to Learn2Reg Challenge."
    International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2020.
    Adopted from:
    https://github.com/yuta-hi/pytorch_similarity
    https://github.com/visva89/pTVreg/blob/master/mutils/My/image_metrics/metric_ngf.m
    """

    def __init__(self,
                 grad_method: str = 'default',
                 gauss_sigma: float = None,
                 gauss_truncate: float = 4.0,
                 eps: Optional[float] = 1e-5,
                 mm_spacing: Optional[Union[int, float, Tuple[int, ...], List[int]]] = None,
                 reduction: Union[LossReduction, str] = LossReduction.MEAN) -> None:
        """
        Args:
            grad_method: {'default', 'sobel', 'prewitt', 'isotropic'}
            type of gradient kernel. Defaults to 'default' (finite difference).
            gauss_sigma: standard deviation from Gaussian kernel. Defaults to None.
            gauss_truncate: trunncate the Gaussian kernel at this number of sd. Defaults to 4.0.
            eps_src: smooth constant for denominator in computing norm of source/moving gradient
            eps_tar: smooth constant for denominator in computing norm of target/fixed gradient
            mm_spacing: pixel spacing of input images
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.
                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
        """
        super().__init__(reduction=LossReduction(reduction).value)

        self.eps = eps

        if isinstance(mm_spacing, (int, float)):
            self.mm_spacing = [mm_spacing] * 3
        if isinstance(mm_spacing, (list, tuple)):
            if len(mm_spacing) == 3:
                self.mm_spacing = mm_spacing
            else:
                raise ValueError(f'expected length 2 spacing, got {mm_spacing}')

        self.grad_method = grad_method
        self.gauss_sigma = _pair(gauss_sigma)
        self.gauss_truncate = gauss_truncate

        self.grad_u_kernel = None
        self.grad_v_kernel = None
        self.grad_w_kernel = None

        self.gauss_kernel_x = None
        self.gauss_kernel_y = None

        self._initialize_params()
        self._freeze_params()

    def _initialize_params(self):
        self._initialize_grad_kernel()
        self._initialize_gauss_kernel()

    def _initialize_grad_kernel(self):
        self.grad_u_kernel = _grad_param(3, self.grad_method, axis=0)
        self.grad_v_kernel = _grad_param(3, self.grad_method, axis=1)
        self.grad_w_kernel = _grad_param(3, self.grad_method, axis=2)

    def _initialize_gauss_kernel(self):
        if self.gauss_sigma[0] is not None:
            self.gauss_kernel_x = _gauss_param(3, self.gauss_sigma[0], self.gauss_truncate)
        if self.gauss_sigma[1] is not None:
            self.gauss_kernel_y = _gauss_param(3, self.gauss_sigma[1], self.gauss_truncate)

    def _check_type_forward(self, x: torch.Tensor):
        if x.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(x.dim()))

    def _freeze_params(self):
        self.grad_u_kernel.requires_grad = False
        self.grad_v_kernel.requires_grad = False
        self.grad_w_kernel.requires_grad = False
        if self.gauss_kernel_x is not None:
            self.gauss_kernel_x.requires_grad = False
        if self.gauss_kernel_y is not None:
            self.gauss_kernel_y.requires_grad = False

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            source: source/moving image, shape should be BCHWD
            target: target/fixed image, shape should be BCHWD
        Returns:
            ngf: normalized gradient field between source and target
        """
        self._check_type_forward(source)
        self._check_type_forward(target)
        self._freeze_params()

        # reshape
        b, c = source.shape[:2]
        spatial_shape = source.shape[2:]

        source = source.view(b * c, 1, *spatial_shape)
        target = target.view(b * c, 1, *spatial_shape)

        # smoothing
        if self.gauss_kernel_x is not None:
            source = spatial_filter_nd(source, self.gauss_kernel_x)
        if self.gauss_kernel_y is not None:
            target = spatial_filter_nd(target, self.gauss_kernel_y)

        # gradient
        src_grad_u = spatial_filter_nd(source, self.grad_u_kernel) * self.mm_spacing[0]
        src_grad_v = spatial_filter_nd(source, self.grad_v_kernel) * self.mm_spacing[1]
        src_grad_w = spatial_filter_nd(source, self.grad_w_kernel) * self.mm_spacing[2]

        tar_grad_u = spatial_filter_nd(target, self.grad_u_kernel) * self.mm_spacing[0]
        tar_grad_v = spatial_filter_nd(target, self.grad_v_kernel) * self.mm_spacing[1]
        tar_grad_w = spatial_filter_nd(target, self.grad_w_kernel) * self.mm_spacing[2]

        if self.eps is None:
            with torch.no_grad():
                self.eps = torch.mean(torch.abs(src_grad_u) + torch.abs(src_grad_v) + torch.abs(src_grad_w))

        # gradient norm
        src_grad_norm = src_grad_u ** 2 + src_grad_v ** 2 + src_grad_w ** 2 + self.eps ** 2
        tar_grad_norm = tar_grad_u ** 2 + tar_grad_v ** 2 + tar_grad_w ** 2 + self.eps ** 2

        # nominator
        product = src_grad_u * tar_grad_u + src_grad_v * tar_grad_v + src_grad_w * tar_grad_w

        # denominator
        denom = src_grad_norm * tar_grad_norm

        # integrator
        ngf = -0.5 * (product ** 2 / denom)

        # reshape back
        ngf = ngf.view(b, c, *spatial_shape)

        # reduction
        if self.reduction == LossReduction.MEAN.value:
            ngf = torch.mean(ngf)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            ngf = torch.sum(ngf)  # sum over batch and channel dims
        elif self.reduction != LossReduction.NONE.value:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return ngf
####################################################################################################



##########################################################
# Loss functions of my project
from utils import *      
class ContextualBilateralLoss3D_local_mm(nn.Module):
    def __init__(self,
                 band_width: float = 0.1,
                 loss_type: str = 'cosine',
                 use_vgg: bool = True,
                 vgg_layer: str = 'relu3_4',
                 weight_sp: float = 1, 
                 vgg_vol_shape=(96,96,96),
                 feat_map: int = 8):
    
        super(ContextualBilateralLoss3D_local_mm, self).__init__()

        assert band_width > 0, 'band_width parameter must be positive.'
        self.band_width = band_width
        self.weight_sp = weight_sp
        self.patch_size = feat_map
        self.loss_type = loss_type
        
        # load feature extractor
        self.resnet1 = vxm.networks.SingleResNet(in_channels=1, resnet_version='base', n_basefilters=16, n_blocks=3).cuda()
        #self.resnet1.load_state_dict(torch.load('./pretrained_model/contrastive_learning/t1_l2/0030.pt')['model_state_dict'])
        self.resnet_dict = torch.load('./pretrained_model/ResNet_pretrained_on_ADNI+NIFD.ckpt')['state_dict']
        new_state_dict = {}
        for key, value in self.resnet_dict.items():
            if 'net.' in key:
                key = key.replace('net.', '')
                key = key.replace('res','resnet.')
            if 'resnet.blocks.3.' in key:
                continue
            if 'resnet.blocks.4.' in key:
                continue
            if 'resnet.blocks.5.' in key:
                continue
            if key in self.resnet1.state_dict():
                new_state_dict[key] = value
        self.resnet_dict['state_dict'] = new_state_dict
        self.resnet1.load_state_dict(self.resnet_dict['state_dict'])
        del self.resnet_dict, new_state_dict
        self.resnet1.eval()

        self.resnet2 = vxm.networks.SingleResNet(in_channels=1, n_outputs=256, num_mlp=2, in_shape=vgg_vol_shape,resnet_version='base', n_basefilters=16, n_blocks=3).cuda() 
        self.resnet2.load_state_dict(torch.load('./pretrained_model/CoBi/t2_l3/0100_final.pt')['model_state_dict'])
        self.resnet2.eval()

        self.grid = compute_meshgrid_3d((1,1,self.patch_size,self.patch_size,self.patch_size)).cuda()
        self.sp_dist = torch.exp(-compute_l2_distance_3d(self.grid, self.grid)) # similarity

    def forward(self, x, y):
        assert x.shape[1] == 1 and y.shape[1] == 1,\
            'VGG3D model takes 1 chennel volumes.'
    
        x, features_x = self.resnet2(x, get_features=True, get_all_features=True)
        x = features_x[2]
        
        y, features_y = self.resnet1(y, get_features=True, get_all_features=True)
        y = features_y[2]
        
        del features_x, features_y
        
        return contextual_bilateral_loss3D_local(x, y, weight_sp=self.weight_sp, band_width=self.band_width, patch_size=self.patch_size, loss_type=self.loss_type, sp_dist=self.sp_dist)


class ContextualBilateralLoss3D_local(nn.Module):
    """
    Creates a criterion that measures the contextual bilateral loss.

    Parameters
    ---
    weight_sp : float, optional
        a balancing weight between spatial and feature loss.
    band_width : int, optional
        a band_width parameter described as :math:`h` in the paper.
    use_vgg : bool, optional
        if you want to use VGG feature, set this `True`.
    vgg_layer : str, optional
        intermidiate layer name for VGG feature.
        Now we support layer names:
            `['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']`
    """

    def __init__(self,
                 band_width: float = 0.1,
                 loss_type: str = 'cosine',
                 use_vgg: bool = True,
                 vgg_layer: str = 'relu3_4',
                 weight_sp: float = 1, 
                 vgg_vol_shape=(64,64,64),
                 feat_map: int = 8):
    
        super(ContextualBilateralLoss3D_local, self).__init__()

        assert band_width > 0, 'band_width parameter must be positive.'
        self.band_width = band_width
        self.weight_sp = weight_sp
        self.patch_size = feat_map
        self.loss_type = loss_type
        
        # Load RESNET
        self.resnet = vxm.networks.SingleResNet(in_channels=1, resnet_version='base', n_basefilters=16, n_blocks=3)
        
        self.resnet_dict = torch.load('./pretrained_model/ResNet_pretrained_on_ADNI+NIFD.ckpt')['state_dict']
        
        new_state_dict = {}
        # Modify keys to match the model's parameter names
        for key, value in self.resnet_dict.items():
         #   # Check if the key is in the model's state_dict
            if 'net.' in key:
                key = key.replace('net.', '')
                key = key.replace('res','resnet.')
            if 'resnet.blocks.3.' in key:
                continue
            if 'resnet.blocks.4.' in key:
                continue
            if 'resnet.blocks.5.' in key:
                continue
            if key in self.resnet.state_dict():
                # Modify key as needed to match the model's parameter names
                new_state_dict[key] = value
        self.resnet_dict['state_dict'] = new_state_dict
        self.resnet.load_state_dict(self.resnet_dict['state_dict'])
        del self.resnet_dict, new_state_dict
        
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.resnet.cuda()
        self.resnet.eval()

        self.grid = compute_meshgrid_3d((1,1,self.patch_size,self.patch_size,self.patch_size)).to('cuda')
        self.sp_dist = torch.exp(-compute_l2_distance_3d(self.grid, self.grid)) # similarity
        
        '''
        ########## resnet FROM MEDAI ##############
        self.resnet = load_function()
        print('nueva pretrained', self.resnet)
        for param in self.resnet.parameters():
            param.requires_grad = False
        '''

    def forward(self, x, y):
        assert x.shape[1] == 1 and y.shape[1] == 1,\
            'RESNET model takes 1 chennel volumes.'
    
        x, features_x = self.resnet(x, get_features=True, get_all_features=True)
        x = features_x[0]

        y, features_y = self.resnet(y, get_features=True, get_all_features=True)
        y = features_y[0]

        del features_x, features_y
        
        return contextual_bilateral_loss3D_local(x, y, weight_sp=self.weight_sp, band_width=self.band_width, patch_size=self.patch_size, loss_type=self.loss_type, sp_dist=self.sp_dist)

def contextual_bilateral_loss3D_local(x: torch.Tensor,
                            y: torch.Tensor,
                            sp_dist: torch.Tensor,
                            weight_sp: float = 0.1,
                            band_width: float = 0.1,
                            loss_type: str = 'cosine', 
                            patch_size: int = 8):
    """
    Computes Contextual Bilateral (CoBi) Loss between x and y,
        proposed in https://arxiv.org/pdf/1905.05169.pdf.

    Parameters
    ---
    x : torch.Tensor
        features of shape (N, C, H, W).
    y : torch.Tensor
        features of shape (N, C, H, W).
    band_width : float, optional
        a band-width parameter used to convert distance to similarity.
        in the paper, this is described as :math:`h`.
    loss_type : str, optional
        a loss type to measure the distance between features.
        Note: `l1` and `l2` frequently raises OOM.

    Returns
    ---
    cx_loss : torch.Tensor
        contextual loss between x and y (Eq (1) in the paper).
    """
    
    assert x.size() == y.size(), 'input tensor must have the same size.'
    
    N, C, H, W, D = x.size()
    
    y_mu = y.mean(dim=(0,2,3,4), keepdim=True)
    x = x - y_mu
    y = y - y_mu
    
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    
    #Need to patch distance due to memory issues
    x_patches = divide_volume(volume=x, patch_size=(patch_size,patch_size,patch_size)) # can be: 8x8x8
    y_patches = divide_volume(volume=y, patch_size=(patch_size,patch_size,patch_size)) # 32x32x32 vols
    
    max_tot = []
    max_tot_idx = []
    
    for i, (x_patch, y_patch) in enumerate(zip(x_patches, y_patches)):
        dist_raw = compute_cosine_similarity_3d(x_patch, y_patch) ##### similarity
        
        dist_combined = (1*dist_raw + 1*sp_dist)/2 ####### similarity
        dist_tilde = compute_relative_distance(1-dist_combined) 
        cx = compute_cx(dist_tilde, band_width)
        max_tot.append(torch.max(cx, dim=1)[0])
    # next line will create tensor where: dim=0-->patches; dim=1-->batches; dim=2-->max_values
    cx_tot = torch.stack(max_tot, dim=0)
    cx_tot = torch.mean(cx_tot, dim=2)  # Eq(1) shape--> n_patches x batch
    cx_loss = torch.mean(-torch.log(cx_tot + 1e-5))  # Eq(5)
    return cx_loss 




class ContextualBilateralLoss3D_local_multilayer(nn.Module):
    """
    Creates a criterion that measures the contextual bilateral loss.

    Parameters
    ---
    weight_sp : float, optional
        a balancing weight between spatial and feature loss.
    band_width : int, optional
        a band_width parameter described as :math:`h` in the paper.
    use_vgg : bool, optional
        if you want to use VGG feature, set this `True`.
    vgg_layer : str, optional
        intermidiate layer name for VGG feature.
        Now we support layer names:
            `['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']`
    """

    def __init__(self,
                 band_width: float = 0.1,
                 loss_type: str = 'cosine',
                 use_vgg: bool = True,
                 vgg_layer: str = 'relu3_4',
                 weight_sp: float = 1, 
                 vgg_vol_shape=(64,64,64),
                 feat_map: int = 8):
    
        super(ContextualBilateralLoss3D_local_multilayer, self).__init__()

        assert band_width > 0, 'band_width parameter must be positive.'
        self.band_width = band_width
        self.weight_sp = weight_sp
        self.patch_size = feat_map
        self.loss_type = loss_type
        
        # Load feature extractor
        self.resnet = vxm.networks.SingleResNet(in_channels=1, resnet_version='base', n_basefilters=16, n_blocks=3)
        self.resnet_dict = torch.load('./pretrained_model/ResNet_pretrained_on_ADNI+NIFD.ckpt')['state_dict']
        new_state_dict = {}
        # Modify keys to match the model's parameter names
        for key, value in self.resnet_dict.items():
         #   # Check if the key is in the model's state_dict
            if 'net.' in key:
                key = key.replace('net.', '')
                key = key.replace('res','resnet.')
            if 'resnet.blocks.3.' in key:
                continue
            if 'resnet.blocks.4.' in key:
                continue
            if 'resnet.blocks.5.' in key:
                continue
            if key in self.resnet.state_dict():
                # Modify key as needed to match the model's parameter names
                new_state_dict[key] = value
        self.resnet_dict['state_dict'] = new_state_dict
        self.resnet.load_state_dict(self.resnet_dict['state_dict'])
        del self.resnet_dict, new_state_dict
        
        for param in self.resnet.parameters():
            #nn.init.normal_(param.data)
            param.requires_grad = False
        self.resnet.cuda()
        self.resnet.eval()


        '''
        #max_mem = torch.cuda.max_memory_allocated() / 2**20 
        #print('memoria del modelo', max_mem) 
        ################ aqui acaba la resnet del ukbiobank ########################
        self.resnet.to('cpu')
        self.resnet.eval()
        '''
        '''
        ########## resnet de internet ##############
        self.resnet = load_function()
        print('nueva pretrained', self.resnet)
        for param in self.resnet.parameters():
            param.requires_grad = False'''

    def forward(self, x, y):
        assert x.shape[1] == 1 and y.shape[1] == 1,\
            'RESNET model takes 1 chennel volumes.'

        x, features_x = self.resnet(x, get_features=True, get_all_features=True)
        x_0 = features_x[0]
        x_1 = features_x[1]
        x_2 = features_x[2] # 16,80,96,112
        x_3 = features_x[3]
        #x_4 = features_x[4]
        
        y, features_y = self.resnet(y, get_features=True, get_all_features=True)
        y_0 = features_x[0]
        y_1 = features_x[1]
        y_2 = features_y[2]
        y_3 = features_y[3]
        #y_4 = features_y[4]

        del x, y, features_x, features_y
        loss0 = contextual_bilateral_loss3D_local(x_0, y_0, weight_sp=self.weight_sp, band_width=self.band_width)
        del x_0, y_0
        loss1 = contextual_bilateral_loss3D_local(x_1, y_1, weight_sp=self.weight_sp, band_width=self.band_width)
        del x_1, y_1
        loss2 = contextual_bilateral_loss3D_local(x_2, y_2, weight_sp=self.weight_sp, band_width=self.band_width)
        del x_2, y_2
        loss3 = contextual_bilateral_loss3D_local(x_3, y_3, weight_sp=self.weight_sp, band_width=self.band_width)
        del x_3, y_3
        #loss4 = contextual_bilateral_loss3D_local(x_4, y_4, weight_sp=self.weight_sp, band_width=self.band_width)
        #del x_4, y_4
        return torch.mean(torch.stack([loss0, loss1, loss2, loss3]))



    

class ContextualLoss3D_patches(nn.Module):
    """
    Creates a criterion that measures the contextual loss.

    Parameters
    ---
    band_width : int, optional a band_width parameter described as :math:`h` in the paper.
    loss_type : similarity metric to use (l1, l2, cosine)
    """

    def __init__(self,
                 band_width: float = 1,
                 loss_type: str = 'cosine',):

        super(ContextualLoss3D_patches, self).__init__()


        assert band_width > 0, 'band_width parameter must be positive.'
        self.band_width = band_width

        # Load the pretrained resnet
        self.resnet = vxm.networks.SingleResNet(in_channels=1, resnet_version='base', n_basefilters=16)
        #self.resnet_dict = torch.load('./pretrained_model/ResNet_pretrained_on_UKBiobank.ckpt')
        self.resnet_dict = torch.load('./pretrained_model/ResNet_pretrained_on_ADNI+NIFD.ckpt')
       
        new_state_dict = {}
        # Modify keys to match the model's parameter names
        for key, value in self.resnet_dict['state_dict'].items():
            # Check if the key is in the model's state_dict
            if 'net.' in key:
                key = key.replace('net.', '')
                key = key.replace('res','resnet.')
            if key in self.resnet.state_dict():
                # Modify key as needed to match the model's parameter names
                new_state_dict[key] = value
        self.resnet_dict['state_dict'] = new_state_dict
        self.resnet.load_state_dict(self.resnet_dict['state_dict'])
        
        del self.resnet_dict, new_state_dict
        

        self.resnet.cuda()
        self.resnet.eval()
        for param in self.resnet.parameters():
            param.requires_grad = False
        print('Feature extractor loaded')

    def forward(self, x, y):
        assert x.shape[1] == 1 and y.shape[1] == 1,\
            'RESNET model takes 1 chennel volumes.'
        
        x, features_x = self.resnet(x, get_features=True, get_all_features=True)
        y, features_y = self.resnet(y, get_features=True, get_all_features=True)
        
        x = features_x[1]
        y = features_y[1]
        
        return contextual_loss3D_patches(x, y, self.band_width)

def contextual_loss3D_patches(x: torch.Tensor,
                y: torch.Tensor,
                band_width: float = 0.5,
                loss_type: str = 'cosine'):
    """
    Computes contextual loss between x and y.
    The most of this code is copied from
        https://gist.github.com/yunjey/3105146c736f9c1055463c33b4c989da.

    Parameters
    ---
    x : torch.Tensor
        features of shape (N, C, H, W).
    y : torch.Tensor
        features of shape (N, C, H, W).
    band_width : float, optional
        a band-width parameter used to convert distance to similarity.
        in the paper, this is described as :math:`h`.
    loss_type : str, optional

    Returns
    ---
    cx_loss : torch.Tensor
        contextual loss between x and y (Eq (1) in the paper)
    """

    assert x.size() == y.size(), 'input tensor must have the same size.'
    #assert loss_type in LOSS_TYPES, f'select a loss type from {LOSS_TYPES}.'

    N, C, H, W, D = x.size()

    # Need to patch distance due to memory issues
    x_patches = divide_volume(volume=x, patch_size=(8,8,8))
    y_patches = divide_volume(volume=y, patch_size=(8,8,8))
    
    max_tot = []
    for i, (x_patch, y_patch) in enumerate(zip(x_patches, y_patches)):
        #print('patch', i, x_patch.shape, y_patch.shape)
        if loss_type == 'cosine':
            dist_raw = compute_cosine_similarity_3d(x_patch, y_patch)
        elif loss_type == 'l1':
            dist_raw = compute_l1_distance_3d(x_patch, y_patch)
        elif loss_type == 'l2':
            dist_raw = compute_l2_distance_3d(x_patch, y_patch)
    
        # dist_raw was a distance matrix (N, h*w*d, h*w*d)
        dist_tilde = compute_relative_distance(dist_raw) # eq(2) normalizing distances
        cx = compute_cx(dist_tilde, band_width) # eq(3,4) distance-->similarit + invariant
        max_tot.append(torch.max(cx, dim=1)[0])
    # next line will create tensor where: dim=0-->batch; dim=1-->max values
    max_cx = torch.stack(max_tot, dim=1)
    cx_tot = torch.mean(max_cx, dim=1)  # Eq(1) 
    cx_loss = torch.mean(-torch.log(cx_tot + 1e-5))  # Eq(5)
    return cx_loss
        


class ContextualLoss3D_patches_local(nn.Module):
    """
    Creates a criterion that measures the contextual loss.

    Parameters
    ---
    band_width : int, optional
        a band_width parameter described as :math:`h` in the paper.
    loss_type : similarity metric to use (l1, l2, cosine)
    """

    def __init__(self,
                 band_width: float = 0.1,
                 loss_type: str = 'cosine',
                 patch_size: int = 8,):

        super(ContextualLoss3D_patches_local, self).__init__()

        assert band_width > 0, 'band_width parameter must be positive.'
        self.band_width = band_width
        self.patch_size = patch_size
        self.loss_type = loss_type
        
        # Load feature extractor
        self.resnet = vxm.networks.SingleResNet(in_channels=1, resnet_version='base', n_basefilters=16, n_blocks=2)
        self.resnet_dict = torch.load('./pretrained_model/ResNet_pretrained_on_ADNI+NIFD.ckpt')['state_dict']

        new_state_dict = {}
        # Modify keys to match the model's parameter names
        for key, value in self.resnet_dict.items():
         #   # Check if the key is in the model's state_dict
            if 'net.' in key:
                key = key.replace('net.', '')
                key = key.replace('res','resnet.')
            if 'resnet.blocks.2.' in key:
                continue
            if 'resnet.blocks.3.' in key:
                continue
            if 'resnet.blocks.4.' in key:
                continue
            if 'resnet.blocks.5.' in key:
                continue
            if key in self.resnet.state_dict():
                # Modify key as needed to match the model's parameter names
                new_state_dict[key] = value

        self.resnet_dict['state_dict'] = new_state_dict
        self.resnet.load_state_dict(self.resnet_dict['state_dict'])

        del self.resnet_dict, new_state_dict

        self.resnet.cuda()
        self.resnet.eval()

        for param in self.resnet.parameters():
            param.requires_grad = False
        
        '''
        self.resnet = vxm.networks.SingleResNet(in_channels=1, resnet_version='base', n_basefilters=16, n_blocks=2)
        #self.resnet.load_state_dict(torch.load('./pretrained_model/pretrained_ukb_state_dict.pt'))
        self.resnet.cuda()
        self.resnet.eval()
        '''
        '''
        ########## resnet from MEDAI ##############
        self.resnet = load_function()
        print('nueva pretrained', self.resnet)
        self.resnet.cuda()
        self.resnet.eval()
        '''

    def forward(self, x, y):
        assert x.shape[1] == 1 and y.shape[1] == 1,\
            'VGG3D model takes 1 chennel volumes.'
        
        x, features_x = self.resnet(x, get_features=True, get_all_features=True)
        y, features_y = self.resnet(y, get_features=True, get_all_features=True)

        x = features_x[2]
        y = features_y[2]

        return contextual_loss3D_patches_local(x, y, band_width=self.band_width, patch_size=self.patch_size, loss_type=self.loss_type)

def contextual_loss3D_patches_local(x: torch.Tensor,
                y: torch.Tensor,
                band_width: float = 0.1,
                patch_size: int = 8,
                loss_type: str = 'cosine'):
    """
    Computes contextual loss between x and y.
    The most of this code is copied from
        https://gist.github.com/yunjey/3105146c736f9c1055463c33b4c989da.

    Parameters
    ---
    x : torch.Tensor
        features of shape (N, C, H, W).
    y : torch.Tensor
        features of shape (N, C, H, W).
    band_width : float, optional
        a band-width parameter used to convert distance to similarity.
        in the paper, this is described as :math:`h`.
    loss_type : str, optional

    Returns
    ---
    cx_loss : torch.Tensor
        contextual loss between x and y (Eq (1) in the paper) evaluated locally
    """

    assert x.size() == y.size(), 'input tensor must have the same size.'
    #assert loss_type in LOSS_TYPES, f'select a loss type from {LOSS_TYPES}.'

    N, C, H, W, D = x.size()
   
    # mean shifting by channel.
    y_mu = y.mean(dim=1, keepdim=True)
    x = x - y_mu
    y = y - y_mu

    # L2 normalization
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)

    # Need to patch distance due to memory issues
    x_patches = divide_volume(volume=x, patch_size=(patch_size,patch_size,patch_size)) # can be: 8x8x8
    y_patches = divide_volume(volume=y, patch_size=(patch_size,patch_size,patch_size)) # 32x32x32 vols
    
    max_tot = []
    for i, (x_patch, y_patch) in enumerate(zip(x_patches, y_patches)):
        dist_raw = compute_cosine_similarity_3d(x_patch, y_patch)
        dist_tilde = compute_relative_distance(dist_raw) # eq(2) normalizing distances
        cx = compute_cx(dist_tilde, band_width) # eq(3,4) distance-->similarit + invariant
        max_tot.append(torch.max(cx, dim=1)[0])
    # next line will create tensor where: dim=0-->patches; dim=1-->batches; dim=2-->max_values
    max_cx = torch.stack(max_tot, dim=0)
    cx_tot = torch.mean(max_cx, dim=2)  # Eq(1) shape--> n_patches x batch
    cx_loss = torch.mean(-torch.log(cx_tot + 1e-5))  # Eq(5)
    return cx_loss 



class Contrastive_loss_resnet(nn.Module):
    def __init__(self,
                 band_width: float = 0.05,
                 feat_map: int = 8):
        
        super(Contrastive_loss_resnet, self).__init__()

        assert band_width > 0, 'band_width parameter must be positive.'
        self.band_width = band_width
        self.patch_size = feat_map
        self.loss = 0.0

    def forward(self, t2_features, t1_features):
        t2_feat3 = t2_features[3]
        t1_feat3 = t1_features[3]
        t2_feat3 = F.normalize(t2_feat3, p=2, dim=1)
        t1_feat3 = F.normalize(t1_feat3, p=2, dim=1)
        t2_feat2 = t2_features[2]
        t1_feat2 = t1_features[2]
        t2_feat2 = F.normalize(t2_feat2, p=2, dim=1)
        t1_feat2 = F.normalize(t1_feat2, p=2, dim=1)
        t2_feat0 = t2_features[0]
        t1_feat0 = t1_features[0]
        t2_feat0 = F.normalize(t2_feat0, p=2, dim=1)
        t1_feat0 = F.normalize(t1_feat0, p=2, dim=1)
        del t2_features, t1_features
        t1_feat_patches3, t2_feat_patches3 = random_select_element(t1_feat3, t2_feat3, 3456)
        t1_feat_patches2, t2_feat_patches2 = random_select_element(t1_feat2, t2_feat2, 11059)
        t1_feat_patches0, t2_feat_patches0 = random_select_element(t1_feat0, t2_feat0, 11118)
        return torch.mean(torch.stack([self.contrastive_loss(t1_feat_patches3, t2_feat_patches3), self.contrastive_loss(t1_feat_patches2, t2_feat_patches2), self.contrastive_loss(t1_feat_patches0, t2_feat_patches0)]))

    def contrastive_loss(self, t1_feat_patches, t2_feat_patches):
        cosine_sim = torch.bmm(t2_feat_patches.transpose(1, 2),t1_feat_patches)
        #print('cosine sim', cosine_sim)
        pos, negs = positives_negatives(cosine_sim)
        #print('1.', pos, negs)
        info_nce1 = (torch.exp((pos)/self.band_width)) / (torch.exp((pos)/self.band_width)+torch.sum(torch.exp((negs)/self.band_width), dim=1))
        #print(info_nce1.shape)
        pos, negs = positives_negatives2(cosine_sim)
        #print('2.', pos, negs)
        info_nce2 = (torch.exp((pos)/self.band_width)) / (torch.exp((pos)/self.band_width)+torch.sum(torch.exp(negs/self.band_width), dim=2))
        #print(info_nce2.shape)
        info_nce = 0.5*(info_nce1+info_nce2)  #batch x k (elements=512=8*8*8)
        #cx_tot = 0.5*torch.mean(info_nce, dim=1) #/len(t2_feat[-1]) = 512 = 8x8x8
        self.cx_loss = torch.mean(-torch.log(info_nce + 1e-5))
        return self.cx_loss

class Contrastive_loss_resnet_patches(nn.Module):
    '''
    Contrastive Loss using only k elements to compare
    '''
    def __init__(self,
                 band_width: float = 0.05,
                 feat_map: int = 8,
                 n_patches: int = 512):
        
        super(Contrastive_loss_resnet_patches, self).__init__()

        assert band_width > 0, 'band_width parameter must be positive.'
        self.band_width = band_width
        self.patch_size = feat_map
        self.loss = 0.0
        self.n_patches = n_patches

    def forward(self, t2_features, t1_features):
        t2_feat0 = t2_features[0]
        t1_feat0 = t1_features[0]
        t2_feat0 = F.normalize(t2_feat0, p=2, dim=1)
        t1_feat0 = F.normalize(t1_feat0, p=2, dim=1)

        del t2_features, t1_features
        
        total_loss = []
        for _ in range(self.n_patches):
            t1_feat_patches0, t2_feat_patches0 = random_select_element(t1_feat0, t2_feat0, 512)
            total_loss.append(self.contrastive_loss(t1_feat_patches0, t2_feat_patches0))
        return torch.mean(torch.stack(total_loss))

    def contrastive_loss(self, t1_feat_patches, t2_feat_patches):
        cosine_sim = torch.bmm(t2_feat_patches.transpose(1, 2),t1_feat_patches)
        
        pos, negs = positives_negatives(cosine_sim)
        
        info_nce1 = (torch.exp((pos)/self.band_width)) / (torch.exp((pos)/self.band_width)+torch.sum(torch.exp((negs)/self.band_width), dim=1))
        
        pos, negs = positives_negatives2(cosine_sim)
        
        info_nce2 = (torch.exp((pos)/self.band_width)) / (torch.exp((pos)/self.band_width)+torch.sum(torch.exp(negs/self.band_width), dim=2))
       
        info_nce = 0.5*(info_nce1+info_nce2)  #batch x k (elements=512=8*8*8)
        #cx_tot = 0.5*torch.mean(info_nce, dim=1) #/len(t2_feat[-1]) = 512 = 8x8x8
        self.cx_loss = torch.mean(-torch.log(info_nce + 1e-5))
        return self.cx_loss



class CXB_resnset(nn.Module):
    '''
    Contextual loss to pre-train RESNETS
    '''
    def __init__(self,
                 band_width: float = 0.1,
                 weight_sp: float = 1, 
                 feat_map: int = 8):
    
        super(CXB_resnset, self).__init__()

        assert band_width > 0, 'band_width parameter must be positive.'
        self.band_width = band_width
        self.weight_sp = weight_sp
        self.patch_size = feat_map

        self.grid = compute_meshgrid_3d((1,1,self.patch_size,self.patch_size,self.patch_size)).to('cuda')
        self.sp_dist = torch.exp(-compute_l2_distance_3d(self.grid, self.grid)) # similarity
    

    def forward(self, t2_features, t1_features):
        
        x = t2_features[3]
        y = t1_features[3]
        y_mu = y.mean(dim=(0,2,3,4), keepdim=True)
        x = x - y_mu
        y = y - y_mu
        N, C, H, W, D = x.size()
        #L2 normalization
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        
        #Need to patch distance due to memory issues
        x_patches = divide_volume(volume=x, patch_size=(self.patch_size,self.patch_size,self.patch_size)) # can be: 8x8x8
        y_patches = divide_volume(volume=y, patch_size=(self.patch_size,self.patch_size,self.patch_size)) # 32x32x32 vols
        
        max_tot = []
        max_tot_idx = []
        
        for i, (x_patch, y_patch) in enumerate(zip(x_patches, y_patches)):
            sim_raw = compute_cosine_similarity_3d(x_patch, y_patch) ##### similarity
            sim_combined = (sim_raw + self.sp_dist)/2 ####### similarity
            dist_tilde = compute_relative_distance(1-sim_combined) # if its the only neighbour, high affinity.
            cx = compute_cx(dist_tilde, self.band_width) # eq(3,4) distance-->similarit + invariant
            max_tot.append(torch.max(cx, dim=1)[0])
        cx_tot = torch.stack(max_tot, dim=0)
        cx_tot = torch.mean(cx_tot, dim=2)  # Eq(1) shape--> n_patches x batch
        cx_loss = torch.mean(-torch.log(cx_tot + 1e-5))  # Eq(5)
        return cx_loss 


class ContrastiveMlp_loss_resnet(nn.Module):
    '''
    Contrastive Loss function to pretrain resnets
    '''
    def __init__(self,
                 band_width: float = 0.05):
    
        super(ContrastiveMlp_loss_resnet, self).__init__()

        assert band_width > 0, 'band_width parameter must be positive.'
        self.band_width = band_width

    def forward(self, t2_features, t1_features):
        '''
        t2_featues --> output of MLP2 of shape batchxn_outputs = 2x128
        t1_featues --> output of MLP1 of shape batchxn_outputs = 2x128
        '''
        t2_feat = F.normalize(t2_features, p=2, dim=1)
        t1_feat = F.normalize(t1_features, p=2, dim=1)
        # the affinit matrix is to compute the positives and negatives in a fast way
        # diagonal elements will be positives, non diagonal will be negatives 
        affin_mat = torch.bmm(t2_feat.unsqueeze(-1), t1_feat.unsqueeze(-1).permute(0,2,1))
        pos, negs = positives_negatives(affin_mat)

        info_nce1 = (torch.exp((pos)/self.band_width)) / (torch.exp((pos)/self.band_width)+torch.sum(torch.exp((negs)/self.band_width), dim=1))
        
        pos, negs = positives_negatives2(affin_mat)
        
        info_nce2 = (torch.exp((pos)/self.band_width)) / (torch.exp((pos)/self.band_width)+torch.sum(torch.exp(negs/self.band_width), dim=2))
        
        info_nce = (info_nce1+info_nce2)  # batch x k (elements=216)
        
        cx_loss = torch.mean(-torch.log(info_nce + 1e-5))

        return cx_loss
