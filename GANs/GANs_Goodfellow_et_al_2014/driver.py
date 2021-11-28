"""Generative Adversarial Networks

Paper: https://arxiv.org/abs/1406.2661

Created on Fri Oct 22 12:27:52 2021

@author: gonzo
"""


import torch
import torch.nn as nn
import torch.optim as optim


from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets


device = "cuda:0" if torch.cuda.is_available() else "cpu"


##############################################################################
#                                                                            #
#                                    Data                                    #
#                                                                            #
##############################################################################

# Use standard FashionMNIST dataset
train_dataset = datasets.FashionMNIST(
    root='./data/train',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.2862, 0.3204),  # Fashion-MNIST
        # transforms.Normalize(0.1530, 0.3042)  # MNIST
    ])
)

test_dataset = datasets.FashionMNIST(
    root='./data/test/',
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(0.1530, 0.3042)  # MNIST
        transforms.Normalize(0.2862, 0.3204),  # Fashion-MNIST
    ])
)

dataset = ConcatDataset([train_dataset, test_dataset])

# # Find the mean and std of the dataset
# loader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=False)

# mean = 0.
# std = 0.
# for images, _ in loader:
#     batch_samples = images.size(0)
#     images = images.view(batch_samples, images.size(1), -1)
#     mean += images.mean(2).sum(0)
#     std += images.std(2).sum(0)

# mean /= len(loader.dataset)
# std /= len(loader.dataset)


##############################################################################
#                                                                            #
#                                   Model                                    #
#                                                                            #
##############################################################################

class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)


##############################################################################
#                                                                            #
#                                Hyperparameters                             #
#                                                                            #
##############################################################################

z_dim = 2 ** 9              # Noise vector
img_dim = 1 * 28 * 28       # [C, H, W]
batch_size = 2 ** 10

generator = Generator(z_dim, img_dim).to(device)
discriminator = Discriminator(img_dim).to(device)

total_params = sum(p.numel() for p in generator.parameters())
total_params += sum(p.numel() for p in discriminator.parameters())
print(f'Number of parameters: {total_params:,}')

fixed_noise = torch.randn((batch_size, z_dim), device=device)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

epochs = 100
lr_G = 3e-4
lr_D = 3e-5

optim_G = optim.Adam(generator.parameters(), lr=lr_G)
optim_D = optim.Adam(discriminator.parameters(), lr=lr_D)
sched_G = CosineAnnealingLR(optim_G, T_max=20, eta_min=0)
sched_D = CosineAnnealingLR(optim_D, T_max=20, eta_min=0)
bce = nn.BCELoss()


##############################################################################
#                                                                            #
#                                  Training                                  #
#                                                                            #
##############################################################################

writer = SummaryWriter("logs/real")
step = 0

for epoch in range(epochs):
    for batch_idx, (real, label) in enumerate(loader):

        real = real.reshape(-1, img_dim).to(device)
        fake = generator(torch.randn((real.shape[0], z_dim), device=device))

        # Maximize Discriminator
        D_real = discriminator(real)
        D_fake = discriminator(fake)
        loss_D_real = bce(D_real, torch.ones_like(D_real))
        loss_D_fake = bce(D_fake, torch.zeros_like(D_fake))
        loss_D = 0.5 * (loss_D_real + loss_D_fake)

        discriminator.zero_grad(set_to_none=True)
        loss_D.backward(retain_graph=True)
        optim_D.step()
        sched_D.step()

        # Minimize Generator
        # D_fake = discriminator(fake)
        loss_G = bce(D_fake, torch.ones_like(D_fake))

        generator.zero_grad(set_to_none=True)
        loss_G.backward(retain_graph=True)
        optim_G.step()
        sched_G.step()

        if batch_idx % 25 == 0:

            print(f"{epoch}.{batch_idx} {loss_D: .3e} {loss_G: .3e}")

            generator.eval()
            with torch.no_grad():
                noise = torch.randn((batch_size, z_dim), device=device)
                fake = generator(noise).reshape(-1, 1, 28, 28)
                img_grid = make_grid(fake, normalize=True)
                writer.add_image("Fake Images", img_grid, global_step=step)
                step += 1
            generator.train()
