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
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import load_CIFAR10
from model import D, G


def train(model, dataloader, optimizer, epochs, device):
    pass


def main(cfg):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img_size = 32*32

    model = {
        'G': G(input_size=cfg['noise_size'], output_size=img_size).to(device),
        'D': D(input_size=img_size).to(device)
        }

    params_g = sum(p.numel() for p in model['G'].parameters())
    params_d = sum(p.numel() for p in model['D'].parameters())
    print(f'{params_g:,} parameters in the generator network')
    print(f'{params_d:,} parameters in the discriminator network')

    dataloader = DataLoader(
        load_CIFAR10(os.path.join('..', 'data')),
        batch_size=cfg['batch_size'],
        shuffle=True,
        drop_last=True,
        pin_memory=torch.cuda.is_available()
        )

    optimizer = {
        'G': Adam(model['G'].parameters(), lr=cfg['lr_g']),
        'D': Adam(model['D'].parameters(), lr=cfg['lr_d'])
        }

    model = train(model, dataloader, optimizer, cfg['epochs'], device)

    # model_dir = os.path.join('..', 'models')
    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir)
    # torch.save(model, os.path.join(model_dir, 'mnist.pt'))

    # plot_manifold(model, 'mnist', device, r0=(-1, 1), r1=(-1, 1))


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