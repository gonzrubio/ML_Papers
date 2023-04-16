#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Common dataset functionality.

Created on Sun Apr 16 15:25:17 2023

@author: gonzalo
"""

import os
import urllib.request
import torch
import scipy.io as sio


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
    else:
        print('Frey Face dataset already downloaded.')

    data = sio.loadmat(file_path)
    data = data['ff'].T.reshape(-1, 1, 28, 20)
    data = torch.from_numpy(data).float()

    return data

