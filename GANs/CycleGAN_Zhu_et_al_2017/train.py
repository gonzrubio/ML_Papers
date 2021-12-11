"""Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.

Paper: https://arxiv.org/abs/1703.10593
Data: https://www.kaggle.com/defileroff/comic-faces-paired-synthetic-v2

Created on Fri Dec 10 19:27:40 2021

@author: gonzr
"""


import os
import torch
import torch.nn as nn
import torch.optim as optim


# from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
# from tqdm import tqdm


from dataset import Face2Comic
from model import Generator
from model import Discriminator


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True


# Data
data_dir = os.getcwd() + '\\data'
train_dataset = Face2Comic(data_dir=data_dir+'\\train', train=False)
val_dataset = Face2Comic(data_dir=data_dir+'\\val', train=False)


# Model
generator = Generator().to(device)
discriminator = Discriminator().to(device)

total_params = sum(p.numel() for p in generator.parameters())
total_params += sum(p.numel() for p in discriminator.parameters())
print(f'Number of parameters: {total_params:,}')


# Hyperparameters
epochs = 500
batch_size = 3
bce = nn.BCEWithLogitsLoss()
mu = 100
L1 = nn.L1Loss()
lr_G = 2e-4
lr_D = 4e-6
betas = (0.5, 0.999)
optim_G = optim.Adam(generator.parameters(), lr=lr_G, betas=betas)
optim_D = optim.Adam(discriminator.parameters(), lr=lr_D, betas=betas)
# sched_G = CosineAnnealingLR(optim_G, T_max=20, eta_min=0)
# sched_D = CosineAnnealingLR(optim_D, T_max=20, eta_min=0)
scaler_D = torch.cuda.amp.GradScaler()
scaler_G = torch.cuda.amp.GradScaler()


# Train loop
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

writer = SummaryWriter()

step = 0
for epoch in range(epochs):
    for batch_idx, (face, comic) in enumerate(train_loader):

        face = face.to(device)
        comic = comic.to(device)
        noise = torch.randn_like(face, device=device)
        fake = generator(face, noise)

        # Maximize Discriminator
        with torch.cuda.amp.autocast():
            D_real = discriminator(face, comic)
            D_fake = discriminator(face, fake)
            loss_D_real = bce(D_real, torch.ones_like(D_real))
            loss_D_fake = bce(D_fake, torch.zeros_like(D_fake))
            loss_D = 0.5 * (loss_D_real + loss_D_fake)

        discriminator.zero_grad(set_to_none=True)
        scaler_D.scale(loss_D).backward(retain_graph=True)
        scaler_D.step(optim_D)
        scaler_D.update()
        # sched_D.step()

        # Minimize Generator
        with torch.cuda.amp.autocast():
            D_fake = discriminator(face, fake)
            loss_L1 = L1(fake, comic)
            loss_G = bce(D_fake, torch.ones_like(D_fake)) + mu * loss_L1

        generator.zero_grad(set_to_none=True)
        scaler_G.scale(loss_G).backward(retain_graph=True)
        scaler_G.step(optim_G)
        scaler_G.update()
        # sched_G.step()

        print(f"{epoch}.{batch_idx} {loss_D: .4e} {loss_G: .4e}")

        if batch_idx % 500 == 0:
            img_grid = make_grid(face, normalize=True)
            writer.add_image("Training: Face", img_grid, global_step=step)
            img_grid = make_grid(comic, normalize=True)
            writer.add_image("Training: True comic", img_grid, global_step=step)
            img_grid = make_grid(fake, normalize=True)
            writer.add_image("Training: Generator comic", img_grid, global_step=step)

            generator.eval()
            with torch.no_grad():
                face, comic = next(iter(val_loader))
                face = face.to(device)
                comic = comic.to(device)
                noise = torch.randn_like(face, device=device)
                fake = generator(face, noise)
                img_grid = make_grid(face, normalize=True)
                writer.add_image("Validation: Face", img_grid, global_step=step)
                img_grid = make_grid(comic, normalize=True)
                writer.add_image("Validation: True comic", img_grid, global_step=step)
                img_grid = make_grid(fake, normalize=True)
                writer.add_image("Validation: Generator comic", img_grid, global_step=step)
                step += 1
            generator.train()

checkpoint_D = {"state_dict": discriminator.state_dict(),
                "optimizer": optim_D.state_dict()}
checkpoint_G = {"state_dict": generator.state_dict(),
                "optimizer": optim_G.state_dict()}

torch.save(checkpoint_D, "discriminator.pt")
torch.save(checkpoint_G, "generator.pt")