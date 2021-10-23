"""Generative Adversarial Networks

Paper: https://arxiv.org/abs/1406.2661

Created on Fri Oct 22 12:27:52 2021

@author: gonzo
"""


import torch
import torch.nn as nn
import torch.optim as optim


from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms, datasets


device = "cuda:0" if torch.cuda.is_available() else "cpu"


##############################################################################
#                                                                            #
#                                    Data                                    #
#                                                                            #
##############################################################################

# Use standard FashionMNIST dataset
train_dataset = datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.2862, 0.3204),
    ])
)

test_dataset = datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.2862, 0.3204),
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

class Generator(nn.Module):
    def __init__(self, in_features=256, img_size=28*28):
        super(Generator, self).__init__()
        self.generate = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.ReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.ReLU(),
            nn.Linear(in_features=in_features, out_features=img_size),
            nn.ReLU(),
            nn.Linear(in_features=img_size, out_features=img_size),
            nn.Tanh()    # To ake sure output range is between +-1
            )

    def forward(self, x):
        return self.generate(x)


class Discriminator(nn.Module):
    def __init__(self, img_size=28*28):
        super(Discriminator, self).__init__()
        dropout_rate = 0.5
        self.classify = nn.Sequential(
            nn.Linear(in_features=img_size, out_features=img_size),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(in_features=img_size, out_features=2*img_size),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(in_features=2*img_size, out_features=2*img_size),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(in_features=2*img_size, out_features=1),
            nn.Sigmoid()   # To output a probability
            )

    def forward(self, x):
        return self.classify(x)


generator = Generator().to(device)
discriminator = Discriminator().to(device)

total_params = sum(p.numel() for p in generator.parameters())
total_params += sum(p.numel() for p in discriminator.parameters())
print(f'Number of parameters: {total_params:,}')

##############################################################################
#                                                                            #
#                                Hyperparameters                             #
#                                                                            #
##############################################################################

z_dim = 2 ** 8              # Noise vector
img_dim = 1 * 28 * 28       # [C, H, W]
batch_size = 2 ** 6

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

epochs = 1000
lr = 3e-4

optim_generator = optim.Adam(generator.parameters(), lr=lr)
optim_discriminator = optim.Adam(discriminator.parameters(), lr=lr)
sched_generator = CosineAnnealingLR(optim_generator, T_max=20, eta_min=0)
sched_discriminator = CosineAnnealingLR(optim_discriminator, T_max=20, eta_min=0)
bce = nn.BCELoss()


##############################################################################
#                                                                            #
#                                  Training                                  #
#                                                                            #
##############################################################################

for epoch in range(epochs):
    for batch_idx, (x, label) in enumerate(loader):

        x = x.reshape(batch_size, -1).to(device)
        noise = torch.randn((batch_size, z_dim), device=device)

        # Maximize Discriminator
        z = generator(noise)
        Dx = discriminator(x)
        Dz = discriminator(z)
        loss_Dx = bce(Dx, torch.ones_like(Dx))
        loss_Dz = bce(Dz, torch.zeros_like(Dz))
        loss_D = 0.5 * (loss_Dx + loss_Dz)

        generator.zero_grad(set_to_none=True)
        loss_D.backward(retain_graph=True)
        optim_discriminator.step()
        sched_discriminator.step()

        # Minimize Generator
        Dz = discriminator(z)
        loss_G = bce(Dz, torch.ones_like(Dz))

        generator.zero_grad(set_to_none=True)
        loss_G.backward(retain_graph=True)
        optim_generator.step()
        sched_generator.step()

    # print(f'loss (MAE): {loss_epoch:.4f}, validation (MAE): {validation:.4f}')





