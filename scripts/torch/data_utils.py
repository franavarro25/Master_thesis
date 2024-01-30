import glob
import os
import numpy as np
import torch
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Union
from pathlib import Path
import csv

from monai.transforms.transform import MapTransform, Transform
from monai.data import DataLoader, Dataset, CacheDataset
from monai.transforms import (  AddChanneld, Compose, LoadImaged, Orientationd, 
                                ResizeWithPadOrCropd, ScaleIntensityRanged,ToTensord, CastToTyped)

# Loading data for OASIS dataset
def load_data(data_dir, **kwargs):
    # list of scan and segm_mask [],[] with names of the files
    data_files = sorted(glob.glob(os.path.join(data_dir, "aligned_norm.nii.gz"), recursive=True))
    scan_files, seg_files = [],[]
    for scan_file in data_files:
        dir = os.path.dirname(scan_file)
        seg_file = os.path.join(dir, "aligned_seg35.nii.gz")
        if os.path.exists(seg_file):
            scan_files.append(scan_file)
            seg_files.append(seg_file)

    # dictionary with the file name
    data_dicts = [{"image": scan_file, "label": seg_file}  for (scan_file, seg_file) in zip(scan_files, seg_files)]
    
    # transformations for the images
    data_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes= 'LIA',),  # reorient to RAS will need post-processing
        ToTensord(keys=["image", "label"],  track_meta=False),
        CastToTyped(keys=["label"], dtype=torch.int64),
        CastToTyped(keys=["image"], dtype=torch.float32) ])

    # build CacheDataset
    dataset = CacheDataset(data=data_dicts, transform=data_transforms, cache_rate=0.1, **kwargs)
    return dataset, data_transforms

# Loading data for OASIS dataset
def data_loader(dataset,batch_size):
    
    training_dataset = dataset[:100]    # will take pairs of two so, 
    validation_dataset = dataset[-40:]  # 50 pairs for training and 20 for val

    training_loader = DataLoader( # dataloader --> dict with keys: image, labels, image_metadata, label_metadata
        training_dataset, batch_size=2*batch_size, shuffle=False, drop_last=True) 
    validation_loader = DataLoader(
        validation_dataset, batch_size=2*batch_size, shuffle=False, drop_last=True)

    train_size = len(training_loader)   # 50 
    val_size = len(validation_loader)   # 46
 
    assert len(training_loader) > 0, 'Could not find any training data.'
    assert len(validation_loader) > 0, 'Could not find any validation data.'

    return training_loader, validation_loader


# Loading data for ADNI dataset
def load_data_multimodal(data_dir, csv_file,**kwargs):
    dataset = CustomPairDataset(data_dir, csv_file) # class defined below
    return dataset

# Loading data for ADNI dataset
def data_loader_multimodal(dataset, batch_size):
    train_dataset = dataset[:100] # will take pairs of two so, 
    val_dataset = dataset[-40:]   # 50 training and 20 validation
    training_loader = DataLoader( # dataloader--> dict with keys: image,labels,image_metadata,label_metadata
                            train_dataset, batch_size=2*batch_size, shuffle=False,
                            drop_last=True) 

    validation_loader = DataLoader( # dataloader--> dict with keys image,labels,image_metadata,label_metadata
                            val_dataset, batch_size=2*batch_size, shuffle=False,
                            drop_last=True) 

    assert len(training_loader) > 0, 'Could not find any training data.'
    assert len(validation_loader) > 0, 'Could not find any validation data.'

    return training_loader, validation_loader 

# Loading data for pre-training
def data_loader_resnet(dataset, batch_size):
    train_dataset = dataset[100:120]  # will only take 1 image at a time (no-pairs)
    val_dataset = dataset[-42:-40] 

    training_loader = DataLoader( # dataloader--> dict with keys image,labels,image_metadata,
                            train_dataset, batch_size=batch_size, shuffle=False,
                            drop_last=True) #label_metadata

    validation_loader = DataLoader( # dataloader--> dict with keys image,labels,image_metadata,
                            val_dataset, batch_size=batch_size, shuffle=False,
                            drop_last=True) #label_metadata
    
    assert len(training_loader) > 0, 'Could not find any training data.'
    assert len(validation_loader) > 0, 'Could not find any validation data.'

    return training_loader, validation_loader 


class CustomPairDataset(Dataset):
    def __init__(self, data_dir, csv_file):
        non_valid_uids = []
        with open(csv_file, 'r', newline='') as file: 
            csv_reader = csv.reader(file)
            for row in csv_reader:
                non_valid_uids.append(row[0]) 
        
        # store all the names of the images in a list
        path = Path(data_dir)
        T1_names, T2_names, mask_names = [],[],[]
        for file in path.rglob('mri1.nii.gz'):
            if file.is_file() and str(file).split('/')[-2] not in non_valid_uids:
                T1_names.append(str(file))
        for file in path.rglob('mri2.nii.gz'):
            if file.is_file() and str(file).split('/')[-2] not in non_valid_uids:
                T2_names.append(str(file))
        for file in path.rglob('aseg.nii.gz'):
            if file.is_file() and str(file).split('/')[-2] not in non_valid_uids:
                mask_names.append(str(file))
        
        self.data_dicts = [{"image1":scan1,"image2":scan2,"seg": seg}   for (scan1,scan2,seg) in zip(T1_names,T2_names,mask_names)]
        
        data_transforms = Compose([
                    LoadImaged(keys=["image1", "image2", "seg"]),
                    AddChanneld(keys=["image1", "image2", "seg"]),
                    ResizeWithPadOrCropd(keys=["image1", "image2","seg"], spatial_size=((96,96,96))), #(160,192,176) or (96,96,96)
                    Orientationd(keys=["image1", "image2", "seg"], axcodes= 'LIA',),  # reorient to RAS will need post-processing
                    ScaleIntensityRanged(keys=["image1", "image2"],
                                         a_min=0.0,
                                         upper=99.9,
                                         b_min=0.0,
                                         b_max=1.0,
                                         clip=False),
                    CastToTyped(keys=["seg"], dtype=torch.int64),
                    CastToTyped(keys=["image1","image2"], dtype=torch.float32),
                    ToTensord(keys=["image1", "image2", "seg"], track_meta=False) ])
        self.dataset = CacheDataset(data=self.data_dicts, transform=data_transforms, cache_rate=0.01)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index, **kwargs):
        return self.dataset[index]



from monai.config import DtypeLike, IndexSelection, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.intensity.array import ScaleIntensityRange
from monai.transforms.transform import MapTransform
from monai.transforms.utils_pytorch_numpy_unification import (clip, percentile,
                                                              where)
class ScaleIntensityRanged(MapTransform):
    """Dictionary-based wrapper of :py:class:`monai.transforms.ScaleIntensityRange`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        a_min: intensity original range min.
        a_max: intensity original range max.
        b_min: intensity target range min.
        b_max: intensity target range max.
        clip: whether to perform clip after scaling.
        dtype: output data type, if None, same as input image. defaults to float32.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = ScaleIntensityRange.backend

    def __init__(
        self,
        keys: KeysCollection,
        a_min: float,
        upper: float,
        b_min: float,
        b_max: float,
        clip: bool = False,
        relative: bool = False,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.a_min = a_min
        self.b_min = b_min
        self.b_max = b_max
        self.upper = upper
        self.clip = clip
        self.relative = relative
        self.dtype = dtype

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]

            a_max: float = percentile(img, self.upper)

            if self.relative:
                b_max = ((self.b_max - self.b_min) *
                         (self.upper / 100.0)) + self.b_min
            else:
                b_max = self.b_max

            scaler = ScaleIntensityRange(a_min=self.a_min,
                                         a_max=a_max,
                                         b_min=self.b_min,
                                         b_max=b_max,
                                         clip=self.clip,
                                         dtype=self.dtype)
            img_sp = scaler(d[key])
            d[key] = clip(img_sp, self.b_min, None)
        return d


class BrainOneHotd(MapTransform):
    def __init__(self,
                 keys: KeysCollection,
                 allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        classes_list = [
            0, 2, 3, 4, 7, 8, 10, 11, 12, 13, 16, 17, 18, 26, 41, 42, 43, 46,
            47, 49, 50, 51, 52, 53, 54, 58, 251, 252, 253, 254, 255
        ]
        num_classes = len(classes_list)

        for key in self.key_iterator(d):
            mask = d[key]
            # the first dimension of mask is channel
            mask = mask.squeeze()
            one_hot = torch.zeros((num_classes, *mask.shape), dtype=torch.uint8)

            for i, c in enumerate(classes_list):
                one_hot[i][mask == c] = 1

            d[key] = one_hot
        return d