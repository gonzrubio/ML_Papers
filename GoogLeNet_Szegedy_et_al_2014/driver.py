"""Going Deeper with Convolutions.

Paper: https://arxiv.org/abs/1409.4842

Created on Wed Oct  6 18:11:57 2021

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


class InceptionBLock(nn.Module):
    def __init__(self, c_in, out_1, red_3, out_3, red_5, out_5, out_1p):
        super(InceptionBLock, self).__init__()

        self.branch_1x1 = nn.Sequential(
            ConvBlock(in_channels=c_in, out_channels=out_1,
                      kernel_size=1, padding='same')
            )

        self.branch_3x3 = nn.Sequential(
            ConvBlock(in_channels=c_in, out_channels=red_3,
                      kernel_size=1, padding='same'),
            ConvBlock(in_channels=red_3, out_channels=out_3,
                      kernel_size=3, padding='same')
            )

        self.branch_5x5 = nn.Sequential(
            ConvBlock(in_channels=c_in, out_channels=red_5,
                      kernel_size=1, padding='same'),
            ConvBlock(in_channels=red_5, out_channels=out_5,
                      kernel_size=5, padding='same')
            )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=c_in, out_channels=out_1p,
                      kernel_size=1, padding='same')
            )

    def forward(self, x):
        x = torch.cat(
            (self.branch_1x1(x),
             self.branch_3x3(x),
             self.branch_5x5(x),
             self.branch_pool(x)),
            dim=1)
        return x


class GoogLeNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1000):
        super(GoogLeNet, self).__init__()

        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=64,
                               kernel_size=3, stride=2, padding=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Sequential(
            ConvBlock(in_channels=64, out_channels=64,
                      kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=64, out_channels=192,
                      kernel_size=3, padding='same'),
            )
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_a = InceptionBLock(c_in=192, out_1=64,
                                          red_3=96, out_3=128,
                                          red_5=16, out_5=32,
                                          out_1p=32)
        self.inception_b = InceptionBLock(c_in=256, out_1=128,
                                          red_3=128, out_3=192,
                                          red_5=32, out_5=96,
                                          out_1p=64)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        pass


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    num_samples = 2**4
    in_channels = 3
    size = 224
    num_classes = 1000

    x_in = torch.randn((num_samples, in_channels, size, size), device=device)
    # inception = InceptionBLock(3, 64, 96, 128, 16, 32, 32).to(device)
    # x_out = inception(x_in)

    model = GoogLeNet(in_channels, num_classes).to(device)
    x_out = model(x_in)

    assert x_out.shape == torch.Size([num_samples, num_classes])
