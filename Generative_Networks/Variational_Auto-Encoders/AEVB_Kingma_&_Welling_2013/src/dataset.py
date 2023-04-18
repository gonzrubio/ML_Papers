#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Common dataset functionality.

Created on Sun Apr 16 15:25:17 2023

@author: gonzalo
"""

import os
import urllib.request
import scipy.io as sio
import torch
import torchvision.datasets as datasets


def load_frey_face_dataset(data_dir):
    """Loads the Frey Face dataset from a .mat file and returns a PyTorch tensor.

    The Frey Face dataset is a collection of 1965 images of human faces. Each
    image is grayscale and has dimensions 28 x 20 pixels.

    Parameters
    ----------
    data_dir : str
        Path to the directory where the dataset is saved or will be saved.

    Returns
    -------
    data : torch.Tensor
        A PyTorch tensor containing the Frey Face dataset. The tensor has shape
        (n_samples, 1, 28, 20), where n_samples is the number of images in the
        dataset.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    file_path = os.path.join(data_dir, 'frey_rawface.mat')
    url = 'https://cs.nyu.edu/~roweis/data/frey_rawface.mat'

    if not os.path.exists(file_path):
        print('Downloading Frey Face dataset...')
        urllib.request.urlretrieve(url, file_path)
        print('Download complete.')

    data = sio.loadmat(file_path)
    data = data['ff'].T.reshape(-1, 1, 28, 20)
    data = torch.from_numpy(data).float()  / 255.0

    return data


def load_mnist_dataset(data_dir):
    """Loads the MNIST dataset and returns a PyTorch tensor.

    The MNIST dataset consists of 28 x 28 grayscale images of handwritten digits
    (0-9). There are 60,000 training images and 10,000 test images.

    Parameters
    ----------
    data_dir : str
        Path to the directory where the dataset is saved or will be saved.

    Returns
    -------
    data : torch.Tensor
        A PyTorch tensor containing the MNIST dataset. The tensor has shape
        (n_samples, 1, 28, 28), where n_samples is the number of images in the
        dataset.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    trainset = datasets.MNIST(root=data_dir, train=True, download=True)
    testset = datasets.MNIST(root=data_dir, train=False, download=True)

    train_data = trainset.data.unsqueeze(1).float() / 255.0
    test_data = testset.data.unsqueeze(1).float() / 255.0

    data = torch.cat([train_data, test_data], dim=0)

    return data
