"""Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.

Paper: https://arxiv.org/abs/1703.10593
Data: https://www.kaggle.com/defileroff/comic-faces-paired-synthetic-v2

Created on Fri Dec 10 19:27:40 2021

@author: gonzr
"""


import albumentations as A
import os
import torch
import torch.nn as nn
import torch.optim as optim


# from torch.optim.lr_scheduler import CosineAnnealingLR
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
# from tqdm import tqdm


from dataset import Face2Comic
from model import Generator, Discriminator


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True


# Data
data_dir = os.getcwd() + '\\data'
transform = A.Compose([
    A.Resize(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
    ToTensorV2()],
    additional_targets={"image0": "image"}
    )

train_dataset = Face2Comic(data_dir=data_dir+'\\train', transform=transform)
val_dataset = Face2Comic(data_dir=data_dir+'\\val', transform=transform)


# Model
G_face2comic = Generator().to(device)        # G(face) -> comic
G_comic2face = Generator().to(device)        # G(comic) -> face

D_face = Discriminator().to(device)
D_comic = Discriminator().to(device)

total_params = sum(p.numel() for p in G_face2comic.parameters())
total_params += sum(p.numel() for p in G_comic2face.parameters())
total_params += sum(p.numel() for p in D_face.parameters())
total_params += sum(p.numel() for p in D_comic.parameters())
print(f'Number of parameters: {total_params:,}')


# Hyperparameters
lr_G = 2e-4
lr_D = 4e-6
betas = (0.5, 0.999)

optim_G = optim.Adam(
    list(G_face2comic.parameters()) + list(G_comic2face.parameters()),
    lr=lr_G, betas=betas)

optim_D = optim.Adam(
    list(D_face.parameters()) + list(D_comic.parameters()),
    lr=lr_D, betas=betas)

# sched_G = CosineAnnealingLR(optim_G, T_max=20, eta_min=0)
# sched_D = CosineAnnealingLR(optim_D, T_max=20, eta_min=0)
scaler_D = torch.cuda.amp.GradScaler()
scaler_G = torch.cuda.amp.GradScaler()

L1 = nn.L1Loss()
MSE = nn.MSELoss()

epochs = 500
batch_size = 2
Lambda = 10


# Train loop
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# writer = SummaryWriter()

step = 0
for epoch in range(epochs):
    for batch_idx, (face, comic) in enumerate(train_loader):

        face = face.to(device)
        comic = comic.to(device)

        # Discriminators D(G(face)) and D(G(comic))
        with torch.cuda.amp.autocast():
            # comic
            fake_comic = G_face2comic(face)
            D_real_comic = D_comic(comic)
            D_fake_comic = D_comic(fake_comic.detach())

            D_real_comic_loss = MSE(D_real_comic,
                                    torch.ones_like(D_real_comic))
            D_fake_comic_loss = MSE(D_fake_comic,
                                    torch.zeros_like(D_fake_comic))

            D_comic_loss = D_real_comic_loss + D_fake_comic_loss

            # face
            fake_face = G_comic2face(comic)
            D_real_face = D_face(face)
            D_fake_face = D_face(fake_face.detach())

            D_real_face_loss = MSE(D_real_face, torch.ones_like(D_real_face))
            D_fake_face_loss = MSE(D_fake_face, torch.zeros_like(D_fake_face))
            D_face_loss = D_real_face_loss + D_fake_face_loss

            # full discriminator objective
            D_loss = (D_comic_loss + D_face_loss)/2

        optim_D.zero_grad(set_to_none=True)
        scaler_D.scale(D_loss).backward(retain_graph=True)
        scaler_D.step(optim_D)
        scaler_D.update()

        # Generators G(face) -> comic and G(comic) -> face
        with torch.cuda.amp.autocast():
            # adversarial generator losses
            D_comic_fake = D_comic(fake_comic)
            D_fake_face = D_face(fake_face)
            G_comic_loss = MSE(D_comic_fake, torch.ones_like(D_comic_fake))
            G_face_loss = MSE(D_fake_face, torch.ones_like(D_fake_face))

            # cycle losses
            cycle_comic = G_face2comic(fake_face)
            cycle_face = G_comic2face(fake_comic)
            cycle_comic_loss = L1(comic, cycle_comic)
            cycle_face_loss = L1(face, cycle_face)

            # identity losses
            identity_comic = G_face2comic(comic)
            identity_face = G_comic2face(face)
            identity_comic_loss = L1(comic, identity_comic)
            identity_face_loss = L1(face, identity_face)

            # full generator objective
            G_loss = (
                G_comic_loss
                + G_face_loss
                + cycle_comic_loss * Lambda
                + cycle_face_loss * Lambda
                + identity_comic_loss * Lambda
                + identity_face_loss * Lambda
            )

        print(f"{epoch}.{batch_idx} {G_loss: .4e}")
