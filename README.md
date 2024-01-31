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

## Spatial Transforms and Integration
- The spatial transform code, found at `voxelmorph.layers.SpatialTransformer`, accepts N-dimensional affine and dense transforms, including linear and nearest neighbor interpolation options. Note that original development of VoxelMorph used `xy` indexing, whereas we are now emphasizing `ij` indexing.

- For the MICCAI2018 version, we integrate the velocity field using `voxelmorph.layers.VecInt`. By default we integrate using scaling and squaring, which we found efficient.




# Data:
The first dataset used in the master thesis was [OASIS1] dataset from (https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md). This dataset can be directly found in NAS storage at TUM, "path/to/dataset". 
The second dataset used in the master thesis was [ADNI] dataset from the TUM group, which is private and cannot be publicly distributed but can be found in NAS storage at TUM "path/to/dataset". Raw data can be found in other folders but the file "ADNI_dataset" has the data direcly available to be used


# Contact:
For any problems or questions do not hesitate to contact me on [franavarro25@gmail.com]
