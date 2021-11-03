"""Improved Training of Wasserstein GANs.

Papers:
    https://arxiv.org/abs/1701.07875
    https://arxiv.org/abs/1704.00028

Created on Tue Oct 26 15:17:08 2021

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


from model import Generator, Critic


device = "cuda:0" if torch.cuda.is_available() else "cpu"


def gradient_penalty(critic, real, fake, device=device):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penalty


def main():

    # Data
    train_dataset = datasets.CIFAR10(
        root='./data/train',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[-0.0163, -0.0347, -0.1056],
                                 std=[0.4045, 0.3987, 0.4020]),
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
        ])
    )

    dataset = ConcatDataset([train_dataset, test_dataset])

    # Hyperparameters
    epochs = 200
    critic_iterations = 5
    lambda_gp = 10
    z_dim = 100
    batch_size = 2 ** 9
    fixed_noise = torch.randn((batch_size, z_dim, 1, 1), device=device)

    generator = Generator().to(device)
    critic = Critic().to(device)

    total_params = sum(p.numel() for p in generator.parameters())
    total_params += sum(p.numel() for p in critic.parameters())
    print(f'Number of parameters: {total_params:,}')

    lr_G = 5e-4
    lr_D = 4e-6
    betas = (0.0, 0.9)

    optim_G = optim.Adam(generator.parameters(), lr=lr_G, betas=betas)
    optim_C = optim.Adam(critic.parameters(), lr=lr_D, betas=betas)
    sched_G = CosineAnnealingLR(optim_G, T_max=20, eta_min=0)
    sched_C = CosineAnnealingLR(optim_C, T_max=20, eta_min=0)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    writer = SummaryWriter("logs/fake")
    step = 0

    for epoch in range(epochs):
        for batch_idx, (real, label) in enumerate(loader):

            # real = real.reshape((-1, 3, 32, 32)).to(device)
            real = real.to(device)

            for iteration in range(critic_iterations):
                noise = torch.randn((real.shape[0], z_dim, 1, 1), device=device)
                fake = generator(noise)
                critic_real = critic(real).reshape(-1)
                critic_fake = critic(fake).reshape(-1)
                gp = gradient_penalty(critic, real, fake, device=device)
                loss_critic = torch.mean(critic_fake) - torch.mean(critic_real)
                loss_critic += lambda_gp * gp
                loss_C = torch.mean(critic_fake) - torch.mean(critic_real)
                critic.zero_grad(set_to_none=True)
                loss_C.backward(retain_graph=True)
                optim_C.step()
                sched_C.step()

            # Minimize Generator
            C_fake = critic(fake)
            loss_G = -torch.mean(C_fake)
            generator.zero_grad(set_to_none=True)
            loss_G.backward()
            optim_G.step()
            sched_G.step()

            if batch_idx % 25 == 0:

                print(f"{epoch}.{batch_idx} {loss_C: .3e} {loss_G: .3e}")

                generator.eval()
                with torch.no_grad():
                    fake = generator(fixed_noise)
                    img_grid = make_grid(fake, normalize=True)
                    writer.add_image("Fake Images", img_grid, global_step=step)
                    step += 1
                generator.train()


if __name__ == '__main__':
    main()
