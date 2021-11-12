"""U-Net: Convolutional Networks for Biomedical Image Segmentation

dataset:https://www.kaggle.com/carlolepelaars/camseq-semantic-segmentation?select=0016E5_07981_L.png
    Julien Fauqueur, Gabriel Brostow, Roberto Cipolla, 
   "Assisted Video Object Labeling By Joint Tracking of Regions and Keypoints", 
   IEEE International Conference on Computer Vision (ICCV'2007) 
   Interactive Computer Vision Workshop. Rio de Janeiro, Brazil, October 2007

Created on Tue Nov  2 18:27:52 2021

@author: gonzr
"""


import numpy as np
import os
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def mean_std(dataset):
    """Return the mean and std of the dataset."""
    
    loader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=False)
    
    mean = 0.
    std = 0.
    for images, _ in loader:
        images = images.to(device)
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
    
    mean /= len(loader.dataset)
    std /= len(loader.dataset)

    return mean, std


one_hot = {
    "64 128 64": 1,     # Animal
    "192 0 128": 2,     # Archway
    "0 128 192": 3,     # Bicyclist
    "0 128 64":4,	    # Bridge
    "128 0 0":5,		# Building
    "64 0 128":6, 	# Car
    "64 0 192":7,	# CartLuggagePram
    "192 128 64":8,	# Child
    "192 192 128":9,	# Column_Pole
    "64 64 128":10,	# Fence
    "128 0 192":11,	# LaneMkgsDriv
    "192 0 64":12,	# LaneMkgsNonDriv
    "128 128 64":13,	# Misc_Text
    "192 0 192":14,	 # MotorcycleScooter
    "128 64 64":15,	 # OtherMoving
    "64 192 128":16,	# ParkingBlock
    "64 64 0":17,		# Pedestrian
    "128 64 128":18,	# Road
    "128 128 192":19,	# RoadShoulder
    "0 0 192":20,		# Sidewalk
    "192 128 128":21,	# SignSymbol
    "128 128 128":22,	# Sky
    "64 128 192":23,	# SUVPickupTruck
    "0 0 64":24,		# TrafficCone
    "0 64 64":25,		# TrafficLight
    "192 64 128":26,	# Train
    "128 128 0":27,	 # Tree
    "192 128 192":28,	# Truck_Bus
    "64 0 64":29,		# Tunnel
    "192 192 0":30,	 # VegetationMisc
    "0 0 0":31,		 # Void
    "64 192 0":32,	 # Wall
    }


def RGB_to_key(channels):
    key = ""
    for elem in channels[:-1]:        
        key += str(elem.item()) + ' '

    return key + str(channels[-1].item())


def transform(image_dir, image_transform, mask_dir, mask_transform):

    images = os.listdir(image_dir)
    # [batch_size, channels, height, width]
    images_tensor = torch.empty((len(images), 3, 224, 224))
    for image_idx, image in enumerate(images):
        image_name = os.path.join(image_dir, image)
        image = Image.open(image_name)
        # image.show()
    
        if image_transform:
            image = image_transform(image)

        images_tensor[image_idx, :] = image

    masks = os.listdir(mask_dir)
    masks_tensor = torch.empty((len(masks), len(one_hot), 224, 224))
    for mask_idx, mask in enumerate(masks):
        mask_name = os.path.join(mask_dir, mask)
        mask = Image.open(mask_name)
        # mask.show()

        if mask_transform:
            mask = np.array(mask_transform(mask))
    
            mask_one_hot = torch.zeros((len(one_hot),
                                        image.shape[-2],
                                        image.shape[-1]))
    
            for row in range(image.shape[-2]):
                for col in range(image.shape[-2]):
    
                    key = RGB_to_key(mask[row, col,:])
                    if key in one_hot:
                        mask_one_hot[one_hot[key] - 1, row, col] = 1.
                    else:
                        mask_one_hot[31 - 1, row, col] = 1.  # void label

        masks_tensor[mask_idx, :] = mask_one_hot
        
    return images_tensor, masks_tensor


class CamSeq01(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.image_transform = image_transform
        self.mask_dir = mask_dir
        self.mask_transform = mask_transform

        self.images, self.masks = transform(self.image_dir,
                                            self.image_transform,
                                            self.mask_dir,
                                            self.mask_transform)
        self.len = self.images.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.images[index], self.masks[index]


if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    image_dir_train = os.getcwd() + '\\data\\train\\image'
    mask_dir_train = os.getcwd() + '\\data\\train\\mask'

    size = (224, 224)

    image_transform_train = transforms.Compose(
        [transforms.Resize(size),
         transforms.ToTensor(),
         transforms.Normalize([0.3158, 0.3349, 0.3497],
                              [0.2301, 0.2595, 0.2577])
             ])

    mask_transform_train = transforms.Compose(
        [transforms.Resize(size),
             ])

    dataset_train = CamSeq01(
        image_dir=image_dir_train,
        mask_dir=mask_dir_train,
        image_transform=image_transform_train,
        mask_transform=mask_transform_train
        )

    mean, std = mean_std(dataset_train)

    print("Mean: ", mean)
    print("Std:", std)

    image_dir_test = os.getcwd() + '\\data\\test\\image'
    mask_dir_test = os.getcwd() + '\\data\\test\\mask'

    dataset_test = CamSeq01(
        image_dir=image_dir_test,
        mask_dir=mask_dir_test,
        image_transform=image_transform_train,
        mask_transform=mask_transform_train
        )

    mean, std = mean_std(dataset_test)

    print("Mean: ", mean)
    print("Std:", std)
