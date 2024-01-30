import nibabel as nib
import matplotlib.pyplot as plt
from data_utils import ScaleIntensityRanged
import numpy as np

img1_ = nib.load('/home/franavarro25/TUM/Master_thesis/dataset/ADNI_remote/ADNI_T1_MNIspace/957745/mri1.nii.gz').get_fdata()
img2_ = nib.load('/home/franavarro25/TUM/Master_thesis/dataset/ADNI_remote/ADNI_T1_MNIspace/1116451/mri1.nii.gz').get_fdata() #878146
img3_ = nib.load('/home/franavarro25/TUM/Master_thesis/dataset/ADNI_remote/ADNI_T1_MNIspace/1230243/mri1.nii.gz').get_fdata()
img1 = {'image': img1_[np.newaxis, np.newaxis, :]}
img2 = {'image': img2_[np.newaxis, np.newaxis, :]}
img3 = {'image': img3_[np.newaxis, np.newaxis, :]}

scalint = ScaleIntensityRanged(keys=["image"],
                                         a_min=0.0,
                                         upper=99.9,
                                         b_min=0.0,
                                         b_max=1.0,
                                         clip=False)

img1 = scalint(img1)['image'].squeeze().detach().numpy()
print('1 ya')
img2 = scalint(img2)['image'].squeeze().detach().numpy()
print('2 ya')
img3 = scalint(img3)['image'].squeeze().detach().numpy()
print('3 ya')
print(type(img1))

# use matplotlib.pyplot.hist to plot the histogram
plt.figure(figsize=(18,6))
fig,ax=plt.subplots(1,3)
_=ax[0].hist(img1_[img1_>0],bins=64)
_=ax[1].hist(img2_[img2_>0],bins=64)
_=ax[2].hist(img3_[img3_>0],bins=64)

plt.figure(figsize=(18,6))
fig,ax=plt.subplots(1,3)
_=ax[0].hist(img1[img1>0],bins=64)
_=ax[1].hist(img2[img2>0],bins=64)
_=ax[2].hist(img3[img3>0],bins=64)

plt.show()