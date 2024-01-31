# Master thesis: Deformable Registration of MRI Data via Contextual Information 
This project is the master thesis from Francisco Navarro Soler (franavarro25@gmail.com)
We have done a study of the loss function baselines used in medical image registration using learning-based methods. 
We have compared the method (Contextual Loss + Spatial Information) to NCC or MI, among other loss functions.
We have used Voxelmorph library, making use of their U-Net architecture, their scale and squaring module, and many other tools for registration. 



# voxelmorph: Learning-Based Image Registration  
**voxelmorph** is a general purpose library for learning-based tools for alignment/registration, and more generally modelling with deformations.


## Parameter choices
All the parameters choices described here are the ones from the original voxelmorph paper. 
### CVPR version
For the CC loss function, we found a reg parameter of 1 to work best. For the MSE loss function, we found 0.01 to work best.

### MICCAI version
For voxelmorph data, we found `image_sigma=0.01` and `prior_lambda=25` to work best.
In the original MICCAI code, the parameters were applied after the scaling of the velocity field. With the newest code, this has been "fixed", with different default parameters reflecting the change. We recommend running the updated code. However, if you'd like to run the very original MICCAI2018 mode, please use `xy` indexing and `use_miccai_int` network option, with MICCAI2018 parameters.


# Train: 
In order to train our 3 main approahes you can use our scripts at ./scripts/torch/
- train_OASIS: train registration network for mono-modality
- train_ADNI: train registration network for multi-modality
- train_ADNI_monomodal_loss: train registration network for multi-modality but using mono-modal loss
- (optional) train_2losses: in case you want to try to train with Intensity based (NCC/MSE) + feature based loss functions (CX)

In all of the training scripts there are functions depending on the type of training you are running: 
1) Weight and biases initialization, you can change parameters or anything you need 
2) load_data() function available in ./scripts/torch/data_utils.py that is different depending if you are training on OASIS or ADNI dataset
3) data_loader() function available in ./scripts/torch/data_utils.py that is adapted for OASIS or ADNI dataset
4) vxm_model() function available in ./scripts/torch/vxm_network.py that defines the network, you can change any parameters that you want. 
5) loss_functions() function available in ./scripts/torch/losses.py where you can choose the similarity loss you want to train with. In losses.py you can also find all the loss functions implemented and available for this project. 
6) Training/evaluation loop: run_epoch() function available in ./scripts/torch/utils.py that gives you the necessary data for training and evaluation
7) Loss computation: depending on the loss function you are using, a different computation might be done and you can check the code in ./scripts/torch/losses.py  Is important to mention that there might be some functions used for the loss function calculation that can be found in ./scripts/torch/utils.py

# Pretrain resnet:
You can try the pre-training strategies at ./scripts/torch/pretrain_resnet_cx.py and ./scripts/torch/pretrain_resnet_mlp.py
You might need to change the loss functions depending on the strategy you want to use. 
You can change anything you want about the loss computation in the losses at ./scripts/torch/losses.py

# Models:
The registration network and the resnet used in the project are available in ./voxelmorph/torch/networks.py

# Data:
The first dataset used in the master thesis was [OASIS1] dataset from (https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md). This dataset can be directly found in NAS storage at TUM, "path/to/dataset". 

The second dataset used in the master thesis was [ADNI] dataset from the TUM group, which is private and cannot be publicly distributed but can be found in NAS storage at TUM "path/to/dataset". Raw data can be found in other folders but the file "ADNI_dataset" has the data direcly available to be used


# Contact:
For any problems or questions do not hesitate to contact me on [franavarro25@gmail.com]
