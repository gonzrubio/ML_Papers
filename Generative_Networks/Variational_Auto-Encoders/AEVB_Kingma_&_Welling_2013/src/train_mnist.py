#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 15:56:42 2023

@author: gonzalo
"""

import os


from dataset import load_mnist_dataset


data_dir = os.path.join('..', 'data')

data = load_mnist_dataset(data_dir)

import matplotlib.pyplot as plt

# Plot a single Frey Face image
plt.imshow(data[0, 0, :, :], cmap='gray')
plt.axis('off')
plt.show()