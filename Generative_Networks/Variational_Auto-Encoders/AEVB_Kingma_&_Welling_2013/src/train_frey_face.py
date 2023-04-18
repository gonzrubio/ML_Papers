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


def train(model, dataloader, optimizer, epochs, device):
    model.train()
    model = model.to(device)

    train_loss = 0
    for batch_idx, x in enumerate(dataloader):
        x = x.to(device)

        x_hat, mu, twicelogvar = model(x)
        BCE = F.binary_cross_entropy(x_hat, x, reduction= 'mean')
        KLD = -0.5 * torch.sum(1 + twicelogvar - mu.pow(2) - twicelogvar.exp())
        loss = BCE + KLD
        train_loss += loss.item()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # TODO print each loss in your style
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    # TODO print total loss in your style
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def main(cfg):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = VAE(
        input_size=28*20,
        hidden_size=cfg['hidden_size'],
        latent_size=cfg['latent_size']
        )

    dataloader = DataLoader(
        load_frey_face_dataset(os.path.join('..', 'data')),
        batch_size=cfg['batch_size'],
        shuffle=True,
        drop_last=False,
        pin_memory=True
        )

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'])

    model = train(model, dataloader, optimizer, cfg['epochs'], device)

    # TODO 10x10 plot learned manifold (sample from unit square)
    plt.imshow(data[0, 0, :, :], cmap='gray')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # TODO parse args, set definition, values and default
    cfg = {
        'batch_size': 100,
        'hidden_size': 200,
        'latent_size': 2,
        'lr': 1e-4,
        'epochs': 100,
        }
    main(cfg)
