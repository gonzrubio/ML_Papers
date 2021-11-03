"""U-Net: Convolutional Networks for Biomedical Image Segmentation

Paper: https://arxiv.org/abs/1505.04597

Created on Tue Nov  2 18:27:52 2021

@author: gonzr
"""


import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super( ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=1, padding=0, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=1, padding=0, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)

        return x


if __name__ == '__main__':
    x = torch.randn((1, 1, 572, 572))
    conv_block = ConvBlock(in_channels=1, out_channels=64)
    x = conv_block(x)
    x.shape
    