#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a VAE on the Frey face dataset and learn the manifold.

Created on Sun Apr 16 15:41:47 2023

@author: gonzalo
"""

import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import load_frey_face_dataset
from model import VAE

import random
def train(model, dataloader, optimizer, epochs, device):
    model.train()

    for epoch in range(epochs):
        for batch_idx, x in enumerate(dataloader):
            x = x.to(device)
            x_hat, mu, twicelogvar = model(x)

            BCE = F.binary_cross_entropy(x_hat, x, reduction= 'sum')
            KLD = torch.sum(1 + twicelogvar - mu.pow(2) - twicelogvar.exp(), dim=-1)
            KLD = torch.sum(-0.5 * KLD)
            batch_loss = BCE + KLD

            optimizer.zero_grad(set_to_none=True)
            batch_loss.backward()
            optimizer.step()

            print(f'{epoch}.{batch_idx}',
                  f"(-'ve)ELBO:{batch_loss.item(): .4f}, "
                  f'BCE:{BCE.item(): .4f}, '
                  f'KLD:{KLD.item(): .2e}')

        if epoch % 500 == 0:
            n = random.randint(0, x_hat.size(0)-1)
            plt.imshow(x[n, 0, :, :].detach().cpu(), cmap='gray')
            plt.axis('off')
            plt.show()            
            plt.imshow(x_hat[n, 0, :, :].detach().cpu(), cmap='gray')
            plt.axis('off')
            plt.show()
            n = random.randint(0, x_hat.size(0)-1)
            plt.imshow(x[n, 0, :, :].detach().cpu(), cmap='gray')
            plt.axis('off')
            plt.show()    
            plt.imshow(x_hat[n, 0, :, :].detach().cpu(), cmap='gray')
            plt.axis('off')
            plt.show()

def main(cfg):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = VAE(
        input_size=28*20,
        hidden_size=cfg['hidden_size'],
        latent_size=cfg['latent_size']
        )

    model = model.to(device)

    dataloader = DataLoader(
        load_frey_face_dataset(os.path.join('..', 'data')),
        batch_size=cfg['batch_size'],
        shuffle=True,
        drop_last=True,
        pin_memory=True
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=cfg['lr'])

    model = train(model, dataloader, optimizer, cfg['epochs'], device)
    print(model)
    # TODO save trained model and edit .gitignore
    # TODO 10x10 plot learned manifold (sample from unit square)
    # plt.imshow(data[0, 0, :, :], cmap='gray')
    # plt.axis('off')
    # plt.show()


if __name__ == '__main__':
    # TODO parse args, set definition, values and default
    cfg = {
        'batch_size': 100,
        'hidden_size': 200,
        'latent_size': 2,
        'lr': 3e-4,
        'epochs': 10000
        }
    main(cfg)
