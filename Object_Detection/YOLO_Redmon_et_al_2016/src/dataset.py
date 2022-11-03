# -*- coding: utf-8 -*-
"""Custom datasets for object detection with YOLO.

Created on Fri Oct 28 13:33:06 2022

@author: gonzr
"""

import numpy as np
import os
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader, default_collate
from utils.bounding_boxes import encode_labels
from utils.plots import plot_batch
from utils.transforms import Transform, AugmentTransform


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
            transform=None,
            S=7, B=2, C=20
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
        :param S: Number of grid cells to split the image for each direction,
        defaults to 7
        :type S: int, optional
        :param B: Number of predicted bounding boxes per gird cell,
        defaults to 2
        :type B: int, optional
        :param C: Number of class labels, defaults to 20
        :type C: int, optional

        """
        self.images_dir = os.path.join(root, 'Images')
        self.annotations_dir = os.path.join(root, 'Annotations')
        with open(os.path.join(root, split + '.txt')) as f:
            self.data = f.read().splitlines()
        self.train = train
        self.transform = AugmentTransform() if self.train else Transform()
        self.S, self.B, self.C = S, B, C

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Read the `idx`-th (image, label) pair."""
        filename = self.data[idx]
        image_path = os.path.join(self.images_dir, filename + '.jpg')
        label_path = os.path.join(self.annotations_dir, filename + '.csv')

        image = Image.open(image_path).convert('RGB')
        labels = torch.from_numpy(
            np.loadtxt(label_path, skiprows=1, delimiter=',')
            )

        image, labels = self.transform(image, labels)
        if self.train:
            labels = encode_labels(labels, self.S, self.B, self.C)

        return image, labels

    def collate_fn(self, batch):
        """Collate function to be passed to the DataLoader.

        :param batch: An iterable of length `len(batch)` containing the sampled
        (image, labels) pairs from __getitem__()
        :type batch: list
        :return: The collated batch. If `train` is `True`, a batched image
        tensor and a batched labels Tensor. Otherwise, a batched image tensor,
        a stacked labels tensor and a tensor of batch indices which maps each
        bounding box to its respective image in the batch.
        :rtype: tuple

        """
        if self.train:
            return default_collate(batch)
        else:
            images = [None] * len(batch)
            labels = [None] * len(batch)
            batch_idx = [None] * len(batch)

            for idx, sample in enumerate(batch):
                images[idx], labels[idx] = sample
                n_bbox = len(labels[idx]) if len(labels[idx].shape) > 1 else 1
                batch_idx[idx] = torch.tensor([idx] * n_bbox)

            images = torch.stack(images, dim=0)
            labels = torch.vstack(labels)
            batch_idx = torch.hstack(batch_idx)

            return (images, labels, batch_idx)


if __name__ == '__main__':

    def voc_detection():
        """Plot collated batches in train and eval mode."""
        for train in [False, True]:
            data = VOCDetection(split='train', train=train)
            kwargs = {
                'batch_size': 5,
                'shuffle': False,
                'collate_fn': data.collate_fn,
                }
            dataloader = DataLoader(data, **kwargs)
            plot_batch(next(iter(dataloader)))

    voc_detection()
    # people_art()
