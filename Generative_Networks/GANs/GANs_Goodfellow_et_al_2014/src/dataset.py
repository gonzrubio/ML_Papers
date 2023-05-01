#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functionality to create the datasets.

Created on Sun Apr 30 17:23:52 2023

@author: gonzalo
"""

import os

import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data_utils


def load_CIFAR10(data_dir):
    """Loads the combined CIFAR-10 train and tests datasets for training.

    The CIFAR-10 dataset consists of 32 x 32 RGB images of 10 different classes
    (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
    There are 50,000 training images and 10,000 test images.

    Parameters
    ----------
    data_dir : str
        Path to the directory where the dataset is saved or will be saved.

    Returns
    -------
    data : torch.Tensor
        A PyTorch tensor containing the CIFAR-10 dataset. The tensor has shape
        (n_samples, 3, 32, 32), where n_samples is the number of images in the
        dataset and 3 corresponds to the RGB channels.

    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.49139968, 0.48215827 ,0.44653124),
            std=(0.24703233, 0.24348505, 0.26158768)
            )
        ])

    trainset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=transform,
        download=True
        )

    testset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        transform=transform,
        download=True
        )

    data = data_utils.ConcatDataset([trainset, testset])

    return data