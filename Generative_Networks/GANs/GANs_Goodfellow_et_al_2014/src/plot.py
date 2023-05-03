#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functionalty to visualize the data.

Created on Mon May  1 21:45:59 2023

@author: gonzalo
"""

import os
import math

import matplotlib.pyplot as plt
import numpy as np


def plot_batch(images, num_cols=10, save=False):
    """Plot a batch of greyscale or RGB images."""
    batch_size, channels, height, width = images.shape
    num_rows = int(math.ceil(batch_size/num_cols))

    # Create a grid of images
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows))
    axes = axes.ravel()

    # Plot each image in the grid
    for i in np.arange(0, batch_size):
        if channels == 1:
            image = np.squeeze(images[i])
        else:
            image = np.transpose(images[i], (1, 2, 0))
        # Rescale the image pixel values to [0, 1]
        image = (image - image.min()) / (image.max() - image.min())
        axes[i].imshow(image, cmap="gray" if channels == 1 else None)
        axes[i].axis("off")

    # Hide any extra axes
    for i in np.arange(batch_size, num_rows * num_cols):
        axes[i].axis("off")

    output_dir = os.path.join('..', 'outputs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.tight_layout()
    if save:
        if channels == 1:
            plt.savefig(os.path.join(output_dir, 'mnist.png'))
        else:
            plt.savefig(os.path.join(output_dir, 'cifar10.png'))
    else:
        plt.show()
    plt.close()
