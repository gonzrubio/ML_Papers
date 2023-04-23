#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functionality to visualize the learned manifolds.

Created on Tue Apr 18 18:18:38 2023

@author: gonzalo
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import torch


def plot_manifold(vae, dataset, device, r0=(-1.5, 1.5), r1=(-1.5, 1.5)):

    if dataset == 'frey':
        w_j = 20
        n = 10
    elif dataset == 'mnist':
        w_j = 28
        n = 20
    else:
        ValueError('Invalid dataset name: {}'.format(dataset))

    w_i = 28
    img = np.zeros((n*w_i, n*w_j))

    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = vae.decoder(z)
            x_hat = x_hat.reshape(w_i, w_j).to('cpu').detach().numpy()
            img[(n-1-i)*w_i:(n-1-i+1)*w_i, j*w_j:(j+1)*w_j] = x_hat

    output_dir = os.path.join('..', 'outputs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join('..', 'outputs', dataset + '.png')

    fig, ax = plt.subplots()
    ax.imshow(img, extent=[*r0, *r1], cmap='gray')
    ax.axis('off')
    fig.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close(fig)
    # fig.savefig(filename, bbox_inches='tight', pad_inches=0)
