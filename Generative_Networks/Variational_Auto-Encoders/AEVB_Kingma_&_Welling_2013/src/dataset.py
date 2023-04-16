#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Common dataset functionality.

Created on Sun Apr 16 15:25:17 2023

@author: gonzalo
"""

import os
import urllib.request


def download_frey_face_dataset(data_dir):
    """
    Downloads the Frey Face dataset and saves it to disk.

    Parameters
    ----------
    data_dir : str
        Path to the directory where the dataset will be saved.

    Returns
    -------
    None
    """

    # URL for downloading the Frey Face dataset
    url = 'https://cs.nyu.edu/~roweis/data/frey_rawface.mat'

    # Create the directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # File path for saving the dataset
    file_path = os.path.join(data_dir, 'frey_rawface.mat')

    # Download the dataset if it doesn't exist
    if not os.path.exists(file_path):
        print('Downloading Frey Face dataset...')
        urllib.request.urlretrieve(url, file_path)
        print('Download complete.')
    else:
        print('Frey Face dataset already downloaded.')
