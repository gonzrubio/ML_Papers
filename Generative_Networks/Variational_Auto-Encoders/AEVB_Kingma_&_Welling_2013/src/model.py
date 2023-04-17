#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The model architecture.

Created on Sun Apr 16 16:28:48 2023

@author: gonzalo
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import load_frey_face_dataset, load_mnist_dataset


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2 * latent_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        mu, twicelogvar = torch.split(x, x.shape[-1] // 2, dim=-1)
        return mu, twicelogvar


class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size=2):
        super().__init__()
        self.latent_size = latent_size
        self.encoder = Encoder(input_size, hidden_size, latent_size)
        self.decoder = Decoder(latent_size, hidden_size, input_size)

    def forward(self, x):
        # approximate posterior distribution parameters
        output_shape = x.shape
        mu, twicelogvar = self.encoder(x.view(x.shape[0], -1))

        # reparameterization trick
        epsilon = torch.randn(x.shape[0], self.latent_size).to(x.device)
        z = mu +  torch.exp(0.5 * twicelogvar) * epsilon

        # reconstructed input
        x = self.decoder(z).view(output_shape)

        return x, mu, twicelogvar


if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # frey face
    batch_size = 100
    data = load_frey_face_dataset(os.path.join('..', 'data')).to(device)
    data = data[:batch_size].to(device)
    input_size = data[0].numel()

    model_frey_face = VAE(input_size=input_size, hidden_size=200).to(device)
    total_params = sum(p.numel() for p in model_frey_face.parameters())
    print(f"Number of parameters for Frey face model: {total_params:,}")

    output, mu, var = model_frey_face(data) 
    assert output.shape == data.shape

    # mnist
    data = load_mnist_dataset(os.path.join('..', 'data')).to(device)
    data = data[:batch_size].to(device)
    input_size = data[0].numel()

    model_mnist = VAE(input_size=input_size, hidden_size=500).to(device)
    total_params = sum(p.numel() for p in model_mnist.parameters())
    print(f"Number of parameters for MNIST model: {total_params:,}")

    output, mu, var = model_mnist(data) 
    assert output.shape == data.shape
