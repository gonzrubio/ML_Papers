"""Image-to-Image Translation with Conditional Adversarial Networks.

Dataset: https://www.kaggle.com/defileroff/comic-faces-paired-synthetic

Created on Mon Nov 22 17:15:42 2021

@author: gonzr
"""


import numpy as np
import os
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()


def mean_std(dataset):
    """Return the mean and std of the dataset."""
    loader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=False)

    mean_inputs = 0.
    std_inputs = 0.
    mean_targets = 0.
    std_targets = 0.

    for inputs, targets in tqdm(loader):

        inputs = inputs.to(DEVICE).view(inputs.size(0), inputs.size(1), -1)
        mean_inputs += inputs.mean(2).sum(0)
        std_inputs += inputs.std(2).sum(0)

        targets = targets.to(DEVICE).view(targets.size(0), inputs.size(1), -1)
        mean_targets += targets.mean(2).sum(0)
        std_targets += targets.std(2).sum(0)

    mean_inputs /= len(loader.dataset)
    std_inputs /= len(loader.dataset)
    mean_targets /= len(loader.dataset)
    std_targets /= len(loader.dataset)

    return (mean_inputs, std_inputs), (mean_targets, std_targets)


class ComicFaces(Dataset):
    """A paired face to comics dataset."""

    def __init__(self, data_dir, transform_faces=None, transform_comics=None):
        super(ComicFaces, self).__init__()
        self.data_dir = data_dir
        self.faces_dir = os.path.join(data_dir, "faces")
        self.faces = os.listdir(self.faces_dir)
        self.comics_dir = os.path.join(data_dir, "comics")
        self.comics = os.listdir(self.comics_dir)
        self.transform_faces = transform_faces
        self.transform_comics = transform_comics
        self.len = len(self.faces)

    def __len__(self):
        """Get the number of samples in the dataset."""
        return self.len

    def __getitem__(self, index):
        """Return the input (face) and the target (comic)."""
        face = Image.open(os.path.join(self.faces_dir, self.faces[index]))
        if self.transform_faces:
            face = self.transform_faces(face)

        comic = Image.open(os.path.join(self.comics_dir, self.comics[index]))
        if self.transform_comics:
            comic = self.transform_comics(comic)

        return face, comic


if __name__ == '__main__':

    data_dir_train = os.getcwd() + '\\data\\train\\'

    # TO DO: Look at albumentations docs and Aladdin and the transforms done
    # in the paper.
    transform_faces = transforms.Compose(
        [transforms.Resize((256, 256)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5129, 0.4136, 0.3671],
                              std=[0.2372, 0.1972, 0.1883])
         ])

    transform_comics = transforms.Compose(
        [transforms.Resize((256, 256)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.4445, 0.3650, 0.3226],
                              std=[0.2594, 0.2051, 0.1840])
         ])

    dataset_train = ComicFaces(
        data_dir=data_dir_train,
        transform_faces=transform_faces,
        transform_comics=transform_comics
        )

    stats_faces, stats_comics = mean_std(dataset_train)

    print(f"Faces: mean = {stats_faces[0]}, std = {stats_faces[1]}")
    print(f"Comics: mean = {stats_comics[0]}, std = {stats_comics[1]}")
