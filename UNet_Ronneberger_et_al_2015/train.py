"""U-Net: Convolutional Networks for Biomedical Image Segmentation

Paper: https://arxiv.org/abs/1505.04597

Created on Sat Nov 13 13:38:17 2021

@author: gonzr
"""


import os

import torch
import torch.nn as nn
import torch.optim as optim


from dataset import CamSeq01
from model import UNet

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms
from torchvision.utils import make_grid


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def train(dataset_train, dataset_val, batch_size, epochs, model, loss_fn, optim):

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    writer = SummaryWriter()

    for epoch in range(epochs):
        for batch_idx, (image, mask) in enumerate(loader_train):

            image = image.to(DEVICE)
            mask = mask.to(DEVICE)

            mask_pred = model(image)
            loss_train = loss_fn(mask_pred, mask)

            loss_train.zero_grad(set_to_none=True)
            loss_train.backward()
            optim.step()
            # sched_D.step()

            model.eval()
            with torch.no_grad():
                for batch_idx, (image, mask) in enumerate(loader_val):
                    n_iter = batch_size * epoch + batch_idx + 1

                    mask_pred = model(image)
                    loss_val = loss_fn(mask_pred, mask)

                    mask_pred_grid = make_grid(mask_pred, normalize=True)
                    mask_grid = make_grid(mask_pred, normalize=True)
                    writer.add_scalar('Loss/train', loss_train, n_iter)
                    writer.add_scalar('Loss/test',loss_val, n_iter)
                    writer.add_image("Predicted Mask", mask_pred_grid, n_iter)
                    writer.add_image("True Mask", mask_grid, n_iter)

            model.train()

            print(f"{epoch}.{batch_idx} {loss_train:.4e} {loss_val:.4e}")


if __name__ == '__main__':

    # Data
    image_dir_train = os.getcwd() + '\\data\\train\\image'
    mask_dir_train = os.getcwd() + '\\data\\train\\mask'
    size = (572, 572)

    image_transform_train = transforms.Compose(
        [transforms.Resize(size),
         transforms.ToTensor(),
         transforms.Normalize([0.3158, 0.3349, 0.3497],
                              [0.2301, 0.2595, 0.2577])
             ])

    mask_transform_train = transforms.Compose(
        [transforms.Resize(size),
             ])

    dataset_train = CamSeq01(
        image_dir=image_dir_train,
        mask_dir=mask_dir_train,
        image_transform=image_transform_train,
        mask_transform=mask_transform_train
        )

    image_dir_test = os.getcwd() + '\\data\\test\\image'
    mask_dir_test = os.getcwd() + '\\data\\test\\mask'

    dataset_val = CamSeq01(
        image_dir=image_dir_test,
        mask_dir=mask_dir_test,
        image_transform=image_transform_train,
        mask_transform=mask_transform_train
        )

    # Model
    model = UNet(in_channels=3, out_channels=32, mirroring=False).to(DEVICE)

    # Hyperparameters
    epochs = 20
    batch_size = 2 ** 2
    loss = nn.BCELoss()
    lr = 5e-4
    optim = optim.Adam(model.parameters(), lr=lr)
    # sched = CosineAnnealingLR(optim, T_max=10, eta_min=0)

    # Train
    train(dataset_train, dataset_val, batch_size, epochs, model, loss, optim)
