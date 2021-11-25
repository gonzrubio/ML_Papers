"""Image-to-Image Translation with Conditional Adversarial Networks.

Paper: https://arxiv.org/abs/1611.07004

Created on Wed Nov 24 21:48:24 2021

@author: gonzr
"""


import torch
import torch.nn as nn
import torch.optim as optim


from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from torchvision.utils import make_grid


from model import Generator, Discriminator


device = "cuda:0" if torch.cuda.is_available() else "cpu"


# TO DO:
# https://github.com/gonzrubio/ML_Papers/blob/main/GANs_Goodfellow_et_al_2014/driver.py

# Data
train_dataset = datasets.CIFAR10(
    root='./data/train',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[-0.0163, -0.0347, -0.1056],
                             std=[0.4045, 0.3987, 0.4020]),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5],
        #                      std=[0.5, 0.5, 0.5]),
    ])
)

test_dataset = datasets.CIFAR10(
    root='./data/test/',
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[-0.0163, -0.0347, -0.1056],
                             std=[0.4045, 0.3987, 0.4020]),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5],
        #                      std=[0.5, 0.5, 0.5])
    ])
)

dataset = ConcatDataset([train_dataset, test_dataset])

# Find the mean and std of the dataset
# loader = DataLoader(dataset, batch_size=128, num_workers=0, shuffle=False)

# mean = torch.tensor((0., 0., 0.))
# std = torch.tensor((0., 0., 0.))
# for images, _ in loader:
#     batch_samples = images.size(0)
#     images = images.view(batch_samples, images.size(1), -1)
#     mean += images.mean(2).sum(0)
#     std += images.std(2).sum(0)

# mean /= len(loader.dataset)
# std /= len(loader.dataset)

# Hyperparameters
epochs = 200
z_dim = 100             # Noise vector
# img_dim = 3 * 32 * 32       # [C, H, W]
batch_size = 2 ** 9
fixed_noise = torch.randn((batch_size, z_dim, 1, 1), device=device)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

total_params = sum(p.numel() for p in generator.parameters())
total_params += sum(p.numel() for p in discriminator.parameters())
print(f'Number of parameters: {total_params:,}')

lr_G = 5e-4
lr_D = 4e-6

optim_G = optim.Adam(generator.parameters(), lr=lr_G)
optim_D = optim.Adam(discriminator.parameters(), lr=lr_D)
# sched_G = CosineAnnealingLR(optim_G, T_max=20, eta_min=0)
# sched_D = CosineAnnealingLR(optim_D, T_max=20, eta_min=0)
bce = nn.BCELoss()

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
writer = SummaryWriter("logs/fake")
step = 0

for epoch in range(epochs):
    for batch_idx, (real, label) in enumerate(loader):

        real = real.reshape((-1, 3, 32, 32)).to(device)
        noise = torch.randn((real.shape[0], z_dim, 1, 1), device=device)
        fake = generator(noise)

        # Maximize Discriminator
        D_real = discriminator(real)
        D_fake = discriminator(fake)
        loss_D_real = bce(D_real, torch.ones_like(D_real))
        loss_D_fake = bce(D_fake, torch.zeros_like(D_fake))
        loss_D = 0.5 * (loss_D_real + loss_D_fake)

        discriminator.zero_grad(set_to_none=True)
        loss_D.backward(retain_graph=True)
        optim_D.step()
        # sched_D.step()

        # Minimize Generator
        D_fake = discriminator(fake)
        loss_G = bce(D_fake, torch.ones_like(D_fake))

        generator.zero_grad(set_to_none=True)
        loss_G.backward(retain_graph=True)
        optim_G.step()
        # sched_G.step()

        if batch_idx % 25 == 0:

            print(f"{epoch}.{batch_idx} {loss_D: .3e} {loss_G: .3e}")

            generator.eval()
            with torch.no_grad():
                fake = generator(fixed_noise)
                img_grid = make_grid(fake, normalize=True)
                writer.add_image("Fake Images", img_grid, global_step=step)
                step += 1
            generator.train()