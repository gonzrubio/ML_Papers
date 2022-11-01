# -*- coding: utf-8 -*-
"""Custom datasets for object detection with YOLO.

Created on Fri Oct 28 13:33:06 2022

@author: gonzr
"""

import numpy as np
import os
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
# from utils.bounding_box import encode_labels, decode_labels
from utils.transforms import Transform, AugmentTransform

# from utils.plots import plot_batch


class VOCDetection(Dataset):
    """Custom Pascal VOC dataset for object detection.

    The original 2007 and 2012 datasets are combined into a single dataset
    which is then randomly divided into train/val/test (70/20/10) splits.

    The combined Pascal VOC dataset has the following directory structure:

        VOC/
        ├─ Images/
        ├  ├── 1.jpg
        ├  ├── 2.jpg
        ├  └── n.jpg
        ├─ Annotations/
        ├  ├── 1.csv
        ├  ├── 2.csv
        ├  └── n.csv
        ├─ train.txt
        ├─ val.txt
        ├─ test.txt

    """

    def __init__(
            self,
            root='../data/VOC',
            split='train',
            train=True,
            transform=None
            ):
        """Parameters.

        :param root: Path to folder storing the dataset,
        defaults to '../data/VOC'
        :type root: str, optional
        :param split: Dataset split, one of 'train', 'val' or 'test',
        defaults to 'train'
        :type split: str, optional
        :param train: Training mode, returns the images and the encoded and
        augmented ground truth labels if True, otherwise it returns the images
        and the raw ground truth labels, defaults to True
        :type train: bool, optional

        """
        self.images_dir = os.path.join(root, 'Images')
        self.annotations_dir = os.path.join(root, 'Annotations')
        with open(os.path.join(root, split + '.txt')) as f:
            self.data = f.read().splitlines()
        self.train = train
        self.transform = AugmentTransform() if self.train else Transform()

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get the (image, label) pair at the `idx` index."""
        filename = self.data[idx]
        image_path = os.path.join(self.images_dir, filename + '.jpg')
        label_path = os.path.join(self.annotations_dir, filename + '.csv')

        image = Image.open(image_path).convert('RGB')
        labels = torch.from_numpy(
            np.loadtxt(label_path, skiprows=1, delimiter=',')
            )

        image, labels = self.transform(image, labels)
        if self.train:
            # labels = encode_labels(labels)
            pass

        return image, labels

    def collate_fn(self, batch):
        """Collate function to be passed to the DataLoader.

        :param batch: An iterable of N (image, labels) pairs from __getitem__()
        :type batch: list
        :return: A single tensor containing the images and a list of
        varying-size tensors with the labels
        :rtype: tuple

        """
        images = [None] * len(batch)
        labels = [None] * len(batch)

        for idx, sample in enumerate(batch):
            images[idx], labels[idx] = sample

        images = torch.stack(images, dim=0)

        return images, labels


def voc_detection():

    # get batch in eval mode and plot it
    data = VOCDetection(split='train', train=False)
    kwargs = {'batch_size': 2, 'shuffle': False, 'collate_fn': data.collate_fn}
    dataloader = DataLoader(data, **kwargs)
    batch = next(iter(dataloader))
    # plot_batch(batch)  # color code class and write class name on bbox

    # get batch in train mode (img, encoded labels)
    # data = YOLO_VOC(split='train', train=True)
    # create data loader and get batch no shuffle so can compare same images
    # dataloader = DataLoader(data, batch_size=64, shuffle=False)
    # batch = next(iter(dataloader))
    # encoded labels to labels, decoding script in utils/bounding_box.py
    # plot_batch(batch)  # color code class and write class name on bbox


if __name__ == '__main__':

    voc_detection()
    # people_art()
