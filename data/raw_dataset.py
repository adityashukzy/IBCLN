import rawpy
import torch
import os.path
import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from skimage.transform import downscale_local_mean

from data.base_dataset import BaseDataset, get_transform
from data.resize_natural_3_dataset import RandomCrop
# from data.image_folder import make_dataset

def make_raw_dataset():
    pass

def get_preprocess_transform(resize_factor=2, ):
    pass

def preprocess_test_data(dataroot, mixed_path, transmission_path):
    mixed_img = rawpy.imread(mixed_path).raw_image
    transmission_img = rawpy.imread(transmission_path).raw_image

    mixed_img = downscale_local_mean(mixed_img, (2,2))
    transmission_img = downscale_local_mean(transmission_img, (2,2))
    
    # convert numpy arrays to torch tensors
    mixed_tensor = torch.from_numpy(mixed_img)
    transmission_tensor = torch.from_numpy(transmission_img)

    i, j, h, w = transforms.RandomCrop.get_params(mixed_tensor, output_size=(512,512))
    mixed_tensor = F.crop(mixed_tensor, i, j, h, w)
    transmission_tensor = F.crop(transmission_tensor, i, j, h, w)

    # convert torch tensors back to numpy arrays
    mixed_img_np = mixed_tensor.numpy()
    transmission_img_np = transmission_tensor.numpy()




class RawDataset(BaseDataset):
    def __init__(self, opt):
        """
            -- if we are running the model in the training phase
                then we are dealing with .npy's that have already been preprocessed
            -- if we are running the model in the inference phase
                then we are dealing with .DNG's that need to first be preprocessed
        """
        BaseDataset.__init__(self, opt)

        self.mixed_dir = os.path.join(opt.dataroot, 'mixed')
        self.transmission_dir = os.path.join(opt.dataroot, 'transmission')
        self.opt = opt

        self.mixed_paths = sorted(make_raw_dataset(self.mixed_dir, opt.max_dataset_size))
        self.transmission_paths = sorted(make_raw_dataset(self.transmission_dir, opt.max_dataset_size))

        self.mixed_size = len(self.mixed_paths)
        self.transmission_size = len(self.transmission_paths)
    
    def __getitem__(self, index):
        """
            Return (index)'th image from dataset.
            If we are in:
                -- training phase: image is .npy, hence preprocessing is already done
                -- testing phase: image is .dng, hence preprocessing is required
        """

        if self.opt.phase == 'test':
            # We need to load images that are in .DNG format from mixed & transmission folders
            # Since testing data is unseen, it needs to be preprocessed first
            
            mixed_path = self.mixed_paths[index]
            # read RAW data into numpy
            mixed_img = rawpy.imread(mixed_path).raw_image
            # downsample by factor of 2
            mixed_img = downscale_local_mean(mixed_img, (2,2))
            # convert to torch tensor
            mixed_tensor = torch.from_numpy(mixed_img)

            if index < len(self.transmission_paths):
                transmission_path = self.transmission_paths[index]
                transmission_img = rawpy.imread(transmission_path).raw_image
                transmission_img = downscale_local_mean(transmission_img, (2,2))
                transmission_tensor = torch.from_numpy(transmission_img)
            else:
                transmission_img = np.zeros_like(mixed_img)
                transmission_tensor = torch.from_numpy(transmission_img)

            is_natural_int = 1
            return {'I': mixed_tensor, 'T': transmission_tensor, 'B_paths': mixed_path, 'isNatural': is_natural_int}

    def __len__(self):
        return self.opt.dataset_size