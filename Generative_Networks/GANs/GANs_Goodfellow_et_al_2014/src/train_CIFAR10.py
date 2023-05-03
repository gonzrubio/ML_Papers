#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generative Adversarial Networks

Paper: https://arxiv.org/abs/1406.2661

Created on Sun Apr 30 17:08:37 2023

@author: gonzalo
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import load_dataset
from model import D, G
from plot import plot_batch


def train(model, dataloader, optimizer, z_dim, epochs, device):

    generator = model['G'].to(device)
    discriminator = model['D'].to(device)

    for epoch in range(epochs):
        for batch_idx, (real, _) in enumerate(dataloader):

            # train the discriminator network
            # minibatch of noise samples {z_1 , ..., z_m } from prior pg(z)
            z = torch.randn(size=(real.shape[0], z_dim), device=device)
            # z = torch.rand(size=(real.shape[0], z_dim), device=device)

            # maximize the discriminator
            fake = generator(z)
            real = real.reshape(fake.shape).to(device)
            D_real = discriminator(real)
            D_fake = discriminator(fake)

            loss_D_real = F.binary_cross_entropy_with_logits(
                input=D_real,
                target=torch.ones_like(D_real),
                reduction='mean'
                )

            loss_D_fake = F.binary_cross_entropy_with_logits(
                input=D_fake,
                target=torch.zeros_like(D_fake),
                reduction='mean'
                )

            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            # loss_D = loss_D_real + loss_D_fake

            discriminator.zero_grad(set_to_none=True)
            loss_D.backward()
            optimizer['D'].step()

            # train the generator network
            # minibatch of noise samples {z_1 , ..., z_m } from prior pg(z)
            z = torch.randn(size=(real.shape[0], z_dim), device=device)
            # z = torch.rand(size=(real.shape[0], z_dim), device=device)

            # minimize the generator
            fake = generator(z)
            D_fake = discriminator(fake)

            loss_G = F.binary_cross_entropy_with_logits(
                input=D_fake,
                target=torch.ones_like(D_fake),
                reduction='mean'
                )

            generator.zero_grad(set_to_none=True)
            loss_G.backward()
            optimizer['G'].step()

            print(f"{epoch}.{batch_idx} {loss_D: .3e} {loss_G: .3e}")

    return generator


def main(cfg):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img_size = 3*32*32

    model = {
        'G': G(input_size=cfg['noise_size'], output_size=img_size),
        'D': D(input_size=img_size)
        }

    params_g = sum(p.numel() for p in model['G'].parameters())
    params_d = sum(p.numel() for p in model['D'].parameters())
    print(f'{params_g:,} parameters in the generator network')
    print(f'{params_d:,} parameters in the discriminator network')

    dataloader = DataLoader(
        load_dataset('CIFAR10', os.path.join('..', 'data')),
        batch_size=cfg['batch_size'],
        shuffle=True,
        drop_last=True,
        pin_memory=torch.cuda.is_available()
        )

    optimizer = {
        'G': Adam(model['G'].parameters(), lr=cfg['lr_g'], betas=(0.5, 0.999)),
        'D': Adam(model['D'].parameters(), lr=cfg['lr_d'], betas=(0.5, 0.999))
        }

    model = train(
        model, dataloader, optimizer, cfg['noise_size'], cfg['epochs'], device
        )

    model_dir = os.path.join('..', 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model, os.path.join(model_dir, 'cifar10.pt'))

    model.eval()
    with torch.no_grad():
        fake = model(torch.randn((100, cfg['noise_size']), device=device))

    plot_batch(fake.reshape(-1, 3, 32, 32).detach().cpu().numpy(), save=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--noise_size', type=int, default=64, help='size of the noise vector')
    parser.add_argument('--lr_generator', type=float, default=3e-4, help='learning rate for the generator')
    parser.add_argument('--lr_discriminator', type=float, default=3e-4, help='learning rate for the discriminator')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train for')
    args = parser.parse_args()

    cfg = {
        'batch_size': args.batch_size,
        'noise_size': args.noise_size,
        'lr_g': args.lr_generator,
        'lr_d': args.lr_discriminator,
        'epochs': args.epochs
        }

    main(cfg)
