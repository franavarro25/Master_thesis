import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from .. import default_unet_features
from . import layers
from .modelio import LoadableModel, store_config_args


class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x):

        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x


class VxmDense(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                'Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)

    def forward(self, source, target, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''
        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x = self.unet_model(x)
        flow_field = self.flow(x)
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)
        preint_flow = pos_flow
        neg_flow = -pos_flow if self.bidir else None
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None
        if not registration:
            return (y_source, y_target, preint_flow) if self.bidir else (y_source, preint_flow)
        else:
            return y_source, pos_flow

##################################################################################################
##################################################################################################
'''
The following code is just an atempt to implement attention blocks in our registration pipeline
but this approach has never been used in the project.
The models are the same as the VxmDense model but with some attention blocks, so all the parameters 
can be checked in the VxmDense Model and the Unet model.
'''
from scripts.torch.utils import *
import voxelmorph as vxm
class VxmDense_t12loss_attention(LoadableModel):
    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False):
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                'Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)

        # load the t1 feature extractor 
        self.resnet = vxm.networks.SingleResNet(in_channels=1, resnet_version='base', n_basefilters=16, n_blocks=3)
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
            if key in self.resnet.state_dict():
                new_state_dict[key] = value
        self.resnet_dict['state_dict'] = new_state_dict
        self.resnet.load_state_dict(self.resnet_dict['state_dict'])
        del self.resnet_dict, new_state_dict
        
        for param in self.resnet.parameters():
            #nn.init.normal_(param.data) #randomly initialize
            param.requires_grad = False
        self.resnet.cuda()
        self.resnet.eval()
        
        
        in_channels = 16
        self.query1_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.key1_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.value1_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.query2_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.key2_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.value2_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.patch_size = 8

        grid = compute_meshgrid_3d((1,1,self.patch_size, self.patch_size, self.patch_size)).to('cuda')
        self.grid = torch.exp(-compute_l2_distance_3d(grid, grid)) # similarity

    def forward(self, source_t2, target_t1, source_t1, target_t2, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''
        batch_size = source_t2.size(0)
        channels_n = 16
        # concatenate inputs and propagate unet
        x = torch.cat([source_t2, target_t1], dim=1)
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_source = self.transformer(source_t1, pos_flow)
        y_target = self.transformer(target_t1, neg_flow) if self.bidir else None

        _, features_source = self.resnet(y_source, get_features=True, get_all_features=True)
        _, features_target = self.resnet(target_t1, get_features=True, get_all_features=True)
        
        # attention t1
        Q1 = self.query1_conv(features_source[0])
        Q1 = F.normalize(Q1, p=2, dim=1)
        K1 = self.key1_conv(features_target[0])
        K1 = F.normalize(K1, p=2, dim=1)
        V1 = self.value1_conv(features_target[0])
        V1 = F.normalize(V1, p=2, dim=1)
        
        q1_patches = divide_volume_attention(volume=Q1, patch_size=(self.patch_size,self.patch_size,self.patch_size))
        k1_patches = divide_volume_attention(volume=K1, patch_size=(self.patch_size,self.patch_size,self.patch_size))
        v1_patches = divide_volume_attention(volume=V1, patch_size=(self.patch_size,self.patch_size,self.patch_size))
        new_feats_t2 = []
        for i, (q1_patch,k1_patch,v1_patch) in enumerate(zip(q1_patches, k1_patches, v1_patches)):
            affin_mat1 = self.softmax(torch.bmm(q1_patch.permute(0,2,1), k1_patch)) ##### similarity 1x512x512
            new_feats_t2.append(torch.bmm(affin_mat1, v1_patch.permute(0,2,1))) # add the value??? 1x512x16

        # attention t2
        Q2 = self.query2_conv(features_target[0])
        Q2 = F.normalize(Q2, p=2, dim=1)
        K2 = self.key2_conv(features_source[0])
        K2 = F.normalize(K2, p=2, dim=1)
        V2 = self.value2_conv(features_source[0])
        V2 = F.normalize(V2, p=2, dim=1)

        del features_source, features_target
        
        q2_patches = divide_volume_attention(volume=Q2, patch_size=(self.patch_size,self.patch_size,self.patch_size))
        k2_patches = divide_volume_attention(volume=K2, patch_size=(self.patch_size,self.patch_size,self.patch_size))
        v2_patches = divide_volume_attention(volume=V2, patch_size=(self.patch_size,self.patch_size,self.patch_size))
        new_feats_t1 = []
        for i, (q2_patch,k2_patch,v2_patch) in enumerate(zip(q2_patches, k2_patches, v2_patches)):
            affin_mat2 = self.softmax(torch.bmm(q2_patch.permute(0,2,1), k2_patch))
            new_feats_t1.append(torch.bmm(affin_mat2, v2_patch.permute(0,2,1))) # 1x512x16

        affin_mat = []
        for t1, t2 in zip(new_feats_t1,new_feats_t2):
            
            affin_mat.append((1 + torch.bmm(t1, t2.permute(0,2,1)))/2 + self.grid)
        return affin_mat, pos_flow

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

class BaseResNet(nn.Module):
    def __init__(self, in_channels: int, 
                n_outputs: int = 3, 
                n_blocks: int = 4, 
                bn_momentum: float = 0.05, 
                n_basefilters: int = 8, 
                resnet_version: str = 'resnet18', 
                bn_track_running_stats: bool = True,
                remain_downsample_steps: int = None,
                no_downsample: bool = False):
        super().__init__()
        self.bn_track_running_stats = bn_track_running_stats
        self.in_channels = in_channels
        self.output_num = n_outputs
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum, kernel_size=5, padding=2, bn_track_running_stats = self.bn_track_running_stats)
        self.pool1 = nn.MaxPool3d(2, stride=2)
        self.no_downsample = no_downsample

        if n_blocks < 2:
            raise ValueError(f"n_blocks must be at least 2, but got {n_blocks}")

        blocks = [
            ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum, 
            bn_track_running_stats = self.bn_track_running_stats, no_downsample = self.no_downsample)
        ]
        n_filters = n_basefilters
        for i in range(n_blocks - 1):
            if remain_downsample_steps and i > remain_downsample_steps:            
                blocks.append(ResBlock(n_filters, 2 * n_filters, bn_momentum=bn_momentum, stride=1, 
                bn_track_running_stats = self.bn_track_running_stats, no_downsample = self.no_downsample))
            else:
                blocks.append(ResBlock(n_filters, 2 * n_filters, bn_momentum=bn_momentum, stride=2, 
                bn_track_running_stats = self.bn_track_running_stats, no_downsample = self.no_downsample))
            n_filters *= 2
        
        self.blocks = nn.ModuleList(blocks)
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        self.resnet_version = resnet_version
        self.resnets = { 'base': None, # choose base
            #'resnet18': generate_model(18, n_input_channels = self.in_channels, 
            #            n_classes = self.output_num, feed_forward = False),             
            #'resnet34': generate_model(34, n_input_channels = self.in_channels, 
            #            n_classes = self.output_num, feed_forward = False), 
            'resnet50': monai.networks.nets.resnet50, 
            'resnet101': monai.networks.nets.resnet101,
            'resnet152': monai.networks.nets.resnet152
        }
        assert resnet_version in self.resnets

    def forward(self, x, get_all_features = False):
        features = []
        if self.resnet_version == 'base':
            out = self.conv1(x)
            features.append(out)
            out = self.pool1(out)
            for block in self.blocks:
                if get_all_features:
                    features.append(out)
                out = block(out)
            if get_all_features:
                features.append(out)
            out = self.global_pool(out)
            if get_all_features:
                features.append(out)
            out = out.view(out.size(0), -1)
            if get_all_features:
                return out, features
            else:
                return out
        
        elif self.resnet_version != 'base':
            resnet_model = self.resnets[self.resnet_version].to(device)
            return resnet_model(x)

class SingleResNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_outputs: int = 3,
        n_blocks: int = 4,
        bn_momentum: float = 0.05,
        n_basefilters: int = 16,
        dropout_rate: float = 0.2,
        resnet_version: str = 'base',
        bn_track_running_stats: bool = True,
        output_features: bool = False,
        remain_downsample_steps: int = None,
        num_mlp: int = 2,
        no_downsample: bool = False,
    ):
        super().__init__()
        self.bn_track_running_stats = bn_track_running_stats
        self.resnet = BaseResNet(
            in_channels=in_channels,
            n_outputs=n_outputs,
            n_blocks=n_blocks,
            bn_momentum=bn_momentum,
            n_basefilters=n_basefilters,
            resnet_version=resnet_version,
            bn_track_running_stats = self.bn_track_running_stats,
            remain_downsample_steps = remain_downsample_steps,
            no_downsample = no_downsample,
        )
        self.n_outputs = n_outputs
        self.dropout = nn.Dropout(p=dropout_rate, inplace=True)
        self.output_features = output_features
        self.num_mlp = num_mlp

        n_filters_out = n_basefilters * (2 ** (n_blocks - 1 +3))
        if resnet_version == 'base':
            if num_mlp == 2:
                self.fc1 = nn.Linear(n_filters_out, n_filters_out, bias=False)
                self.bn = nn.BatchNorm1d(n_filters_out, track_running_stats = self.bn_track_running_stats)
                self.relu = nn.ReLU(inplace=True)

                self.fc2 = nn.Linear(n_filters_out, n_outputs)
            
            elif num_mlp == 3:
                self.fc1 = nn.Linear(n_filters_out, n_filters_out, bias=False)
                self.bn1 = nn.BatchNorm1d(n_filters_out, track_running_stats = self.bn_track_running_stats)
                self.relu1 = nn.ReLU(inplace=True)
                self.fc2 = nn.Linear(n_filters_out, n_filters_out // 2, bias=True)
                self.bn2 = nn.BatchNorm1d(n_filters_out // 2, track_running_stats = self.bn_track_running_stats)
                self.relu2 = nn.ReLU(inplace=True)

                self.fc3 = nn.Linear(n_filters_out // 2, n_outputs)

        else:
            self.fc1 = nn.Linear(512, n_filters_out, bias=False)
            self.bn = nn.BatchNorm1d(n_filters_out, track_running_stats = self.bn_track_running_stats)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(n_filters_out, n_outputs)


    def forward(self, inputs, get_features: bool = False, get_all_features: bool = False):
        if get_features is None:
            out = self.resnet(inputs)
            features = out
            out = self.dropout(out)
            if self.num_mlp == 2:
                out = self.fc1(out)
                out = self.relu(self.bn(out))
                out = self.fc2(out)
            elif self.num_mlp == 3:
                out = self.fc1(out)
                out = self.relu1(self.bn1(out))
                out = self.fc2(out)
                out = self.relu2(self.bn2(out))
                out = self.fc3(out)
            return features, out
        else:
            if get_features and get_all_features:
                out, features = self.resnet(inputs, get_all_features = get_all_features)
                return out, features
            else: 
                out = self.resnet(inputs)

            
            if not get_features:
                out = self.dropout(out)
                if self.num_mlp == 2:
                    out = self.fc1(out)
                    out = self.relu(self.bn(out))
                    out = self.fc2(out)
                elif self.num_mlp == 3:
                    out = self.fc1(out)
                    out = self.relu1(self.bn1(out))
                    out = self.fc2(out)
                    out = self.relu2(self.bn2(out))
                    out = self.fc3(out)
            return out
##################################################################################################
##################################################################################################
