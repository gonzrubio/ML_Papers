"""Deep Residual Learning for Image Recognition.

Paper: https://arxiv.org/abs/1512.03385

Created on Tue Oct 12 20:09:50 2021

@author: gonzo
"""


import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batch_norm(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(ResBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=1, padding='same')
        self.conv2 = ConvBlock(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, stride=stride, padding=1)
        self.conv3 = ConvBlock(in_channels=in_channels, out_channels=4*in_channels,
                               kernel_size=1, padding='same')
        self.downsample = downsample

    def forward(self, x):

        identity = self.downsample(x) if self.downsample else x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x + identity


class ResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1000, num_layers=50):
        assert num_layers in [50, 101, 152]
        if num_layers == 50:
            num_layers = [3, 4, 6, 3]
        elif num_layers == 101:
            num_layers = [3, 4, 23, 3]
        else:
            num_layers = [3, 8, 36, 3]

        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=64,
                               kernel_size=7, stride=2, padding=3)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x = 

if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    num_samples = 2**5
    in_channels = 3
    size = 224
    num_classes = 1000

    x_in = torch.randn((num_samples, in_channels, size, size), device=device)
    model = ResNet(in_channels, num_classes).to(device)
    x_out = model(x_in)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters for ligand network: {total_params:,}")

    assert x_out.shape == torch.Size([num_samples, num_classes])
    