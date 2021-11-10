"""U-Net: Convolutional Networks for Biomedical Image Segmentation

dataset:https://www.kaggle.com/carlolepelaars/camseq-semantic-segmentation?select=0016E5_07981_L.png
    Julien Fauqueur, Gabriel Brostow, Roberto Cipolla, 
   "Assisted Video Object Labeling By Joint Tracking of Regions and Keypoints", 
   IEEE International Conference on Computer Vision (ICCV'2007) 
   Interactive Computer Vision Workshop. Rio de Janeiro, Brazil, October 2007

Created on Tue Nov  2 18:27:52 2021

@author: gonzr
"""

import matplotlib.pyplot as plt
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CamSeq01(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.images = os.listdir(image_dir)
        self.mask_dir = mask_dir
        self.masks = os.listdir(mask_dir)
        self.len = len(self.images)
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image_name = os.path.join(self.image_dir, self.images[index])
        mask_name = os.path.join(self.mask_dir, self.images[index])
        image = Image.open(image_name)
        mask = Image.open(mask_name.replace(".png", "_L.png"))

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask


if __name__ == '__main__':

    image_dir_train = os.getcwd() + '\\data\\train\\image'
    mask_dir_train = os.getcwd() + '\\data\\train\\mask'

    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])])

    dataset_train = CamSeq01(
        image_dir=image_dir_train,
        mask_dir=mask_dir_train,
        transform=transform
        )

    print("Mean: ", dataset_train[0][0].mean(dim=1).mean(dim=1))
    print("Std:", dataset_train[0][0].std(dim=1).std(dim=1))


