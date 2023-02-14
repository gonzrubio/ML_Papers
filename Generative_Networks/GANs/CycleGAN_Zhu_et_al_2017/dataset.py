"""Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.

Paper: https://arxiv.org/abs/1703.10593
Data: https://www.kaggle.com/defileroff/comic-faces-paired-synthetic-v2

Created on Sun Dec  5 18:53:18 2021

@author: gonzr
"""

import albumentations as A
import numpy as np
import os
import torch


from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()


def mean_std(dataset):
    """Return the mean and std of the dataset."""
    loader = DataLoader(dataset, batch_size=128, num_workers=0, shuffle=False)

    mean_inputs = 0.
    std_inputs = 0.
    mean_targets = 0.
    std_targets = 0.

    for inputs, targets in tqdm(loader):

        inputs = inputs.to(DEVICE).view(inputs.size(0), inputs.size(1), -1)
        mean_inputs += inputs.mean(2).sum(0)
        std_inputs += inputs.std(2).sum(0)

        targets = targets.to(DEVICE).view(targets.size(0), inputs.size(1), -1).float()
        mean_targets += targets.mean(2).sum(0)
        std_targets += targets.std(2).sum(0)

    mean_inputs /= len(loader.dataset)
    std_inputs /= len(loader.dataset)
    mean_targets /= len(loader.dataset)
    std_targets /= len(loader.dataset)

    return (mean_inputs, std_inputs), (mean_targets, std_targets)


class Face2Comic(Dataset):
    """A paired face-to-comics dataset."""

    def __init__(self, data_dir, transform):
        super(Face2Comic, self).__init__()
        self.data_dir = data_dir

        self.faces_dir = os.path.join(data_dir, "faces")
        self.faces = os.listdir(self.faces_dir)

        self.comics_dir = os.path.join(data_dir, "comics")
        self.comics = os.listdir(self.comics_dir)

        self.len = len(self.faces)
        self.transform = transform

    def __len__(self):
        """Get the number of samples in the dataset."""
        return self.len

    def __getitem__(self, index):
        """Return the transformed input (face) and target (comic)."""
        face = Image.open(os.path.join(self.faces_dir, self.faces[index]))
        comic = Image.open(os.path.join(self.comics_dir, self.comics[index]))

        augmentations = self.transform(image=np.array(face),
                                       image0=np.array(comic))

        return augmentations["image"], augmentations["image0"]


if __name__ == '__main__':

    data_dir_train = os.getcwd() + '\\data\\train\\'
    transform = A.Compose([
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2()],
        additional_targets={"image0": "image"}
        )

    dataset_train = Face2Comic(data_dir=data_dir_train, transform=transform)

    stats_faces, stats_comics = mean_std(dataset_train)

    print(f"Faces: mean = {stats_faces[0]}, std = {stats_faces[1]}")
    print(f"Comics: mean = {stats_comics[0]}, std = {stats_comics[1]}")

    data_dir_val = os.getcwd() + '\\data\\val\\'
    dataset_val = Face2Comic(data_dir=data_dir_val, transform=transform)

    stats_faces, stats_comics = mean_std(dataset_val)

    print(f"Faces: mean = {stats_faces[0]}, std = {stats_faces[1]}")
    print(f"Comics: mean = {stats_comics[0]}, std = {stats_comics[1]}")
