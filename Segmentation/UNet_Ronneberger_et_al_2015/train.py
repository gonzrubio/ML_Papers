"""U-Net: Convolutional Networks for Biomedical Image Segmentation

Paper: https://arxiv.org/abs/1505.04597

Created on Sat Nov 13 13:38:17 2021

@author: gonzr

TO DO:
    - Assign class indices for labels instead of one hot encoding to allow for
    optimized memory and computation.
    - Compute class weights.
    - Data augmentation (more transforms, apply transforms in to getter not
    in the contructor).
"""


import os

import torch
import torch.nn as nn
import torch.optim as optim


from dataset import CamSeq01
from model import UNet

# from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision.utils import make_grid


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2


def train(dataset_train, dataset_val, batch_size, epochs, model, loss_fn, optim):

    loader_train = DataLoader(dataset_train, batch_size=batch_size,
                              shuffle=True, num_workers=NUM_WORKERS)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    writer = SummaryWriter("tensorboard/")

    for epoch in range(epochs):
        for batch_idx, (image, mask) in enumerate(loader_train):

            image = image.to(DEVICE)
            mask = mask.to(DEVICE)
            # Convert to class index and reshape to [batch_size, height, width]

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
    path = os.getcwd() + "\\data"
    dataset_train = torch.load(path + "\\train\\dataset_train.pt")
    dataset_val = torch.load(path + "\\test\\dataset_test.pt")

    # Model
    model = UNet(in_channels=3, out_channels=32, mirroring=False).to(DEVICE)

    # Hyperparameters
    epochs = 20
    batch_size = 1
    loss = nn.CrossEntropyLoss()
    lr = 5e-4
    optim = optim.Adam(model.parameters(), lr=lr)
    # sched = CosineAnnealingLR(optim, T_max=10, eta_min=0)

    # Train
    train(dataset_train, dataset_val, batch_size, epochs, model, loss, optim)
