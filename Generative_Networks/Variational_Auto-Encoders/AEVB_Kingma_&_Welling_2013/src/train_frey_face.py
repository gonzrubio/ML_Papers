#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a VAE on the Frey face dataset and learn the manifold.

Created on Sun Apr 16 15:41:47 2023

@author: gonzalo
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import load_frey_face_dataset
from model import VAE
from visualize_manifold import plot_manifold


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

        if epoch % 500 == 0:
            print(f'Epoch {epoch}:',
                  f"(-'ve)ELBO:{batch_loss.item(): .4f}, "
                  f'BCE:{BCE.item(): .4f}, '
                  f'KLD:{KLD.item(): .2e}')

    print(f'Epoch {epoch}:',
          f"(-'ve)ELBO:{batch_loss.item(): .4f}, "
          f'BCE:{BCE.item(): .4f}, '
          f'KLD:{KLD.item(): .2e}')

    return model


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

    model_dir = os.path.join('..', 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model, os.path.join(model_dir, 'frey.pt'))

    plot_manifold(model, 'frey', device, r0=(-1, 1), r1=(-1, 1))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100, help='batch size for training')
    parser.add_argument('--hidden_size', type=int, default=200, help='size of the hidden layer')
    parser.add_argument('--latent_size', type=int, default=2, help='size of the latent space')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=2000, help='number of epochs to train for')
    args = parser.parse_args()

    cfg = {
        'batch_size': args.batch_size,
        'hidden_size': args.hidden_size,
        'latent_size': args.latent_size,
        'lr': args.lr,
        'epochs': args.epochs
    }

    main(cfg)
