"""Unsupervised Representation Learning with Deep Convolutional GANs.

Paper: https://arxiv.org/abs/1511.06434

Created on Mon Oct 25 13:23:42 2021

@author: gonzo
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


def main():

    # Data
    train_dataset = datasets.CIFAR10(
        root='./data/train',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])
    )

    test_dataset = datasets.CIFAR10(
        root='./data/test/',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
    )

    dataset = ConcatDataset([train_dataset, test_dataset])

    # Hyperparameters
    epochs = 100
    z_dim = 100             # Noise vector
    # img_dim = 3 * 32 * 32       # [C, H, W]
    batch_size = 2 ** 7
    fixed_noise = torch.randn((batch_size, z_dim, 1, 1), device=device)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    total_params = sum(p.numel() for p in generator.parameters())
    total_params += sum(p.numel() for p in discriminator.parameters())
    print(f'Number of parameters: {total_params:,}')

    lr_G = 9e-5
    lr_D = 4e-6

    optim_G = optim.Adam(generator.parameters(), lr=lr_G)
    optim_D = optim.Adam(discriminator.parameters(), lr=lr_D)
    sched_G = CosineAnnealingLR(optim_G, T_max=20, eta_min=0)
    sched_D = CosineAnnealingLR(optim_D, T_max=20, eta_min=0)
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
            sched_D.step()

            # Minimize Generator
            D_fake = discriminator(fake)
            loss_G = bce(D_fake, torch.ones_like(D_fake))

            generator.zero_grad(set_to_none=True)
            loss_G.backward(retain_graph=True)
            optim_G.step()
            sched_G.step()

            if batch_idx % 25 == 0:

                print(f"{epoch}.{batch_idx} {loss_D: .3e} {loss_G: .3e}")

                generator.eval()
                with torch.no_grad():
                    fake = generator(noise)
                    img_grid = make_grid(fake, normalize=True)
                    writer.add_image("Fake Images", img_grid, global_step=step)
                    step += 1
                generator.train()


if __name__ == '__main__':
    main()
