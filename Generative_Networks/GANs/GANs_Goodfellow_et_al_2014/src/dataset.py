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


def load_dataset(dataset_name, data_dir):
    """Loads the CIFAR-10 or MNIST dataset for training.

    The CIFAR-10 dataset consists of 32 x 32 RGB images of 10 different classes
    (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
    There are 50,000 training images and 10,000 test images.

    The MNIST dataset consists of 28 x 28 grayscale images of handwritten digits
    from 0 to 9. There are 60,000 training images and 10,000 test images.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load. Valid values are "CIFAR10" or "MNIST".
    data_dir : str
        Path to the directory where the dataset is saved or will be saved.

    Returns
    -------
    data : torch.Tensor
        A PyTorch tensor containing the dataset. The tensor has shape
        (n_samples, channels, height, width), where n_samples is the number of
        images in the dataset, channels is the number of color channels (3 for
        CIFAR-10, 1 for MNIST), and height and width are the dimensions of the
        images.

    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if dataset_name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(.5, .5, .5),
                std=(.5, .5, .5)
                # mean=(0.49139968, 0.48215827 ,0.44653124),
                # std=(0.24703233, 0.24348505, 0.26158768)
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

    elif dataset_name == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
            ])

        trainset = datasets.MNIST(
            root=data_dir,
            train=True,
            transform=transform,
            download=True
            )

        testset = datasets.MNIST(
            root=data_dir,
            train=False,
            transform=transform,
            download=True
            )

    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    data = data_utils.ConcatDataset([trainset, testset])

    return data
