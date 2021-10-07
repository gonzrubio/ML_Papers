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


class GoogLeNet(nn.Module):
    pass


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    num_samples = 2**4
    in_channels = 3
    size = 224
    num_classes = 1000

    x_in = torch.randn((num_samples, in_channels, size, size), device=device)
    model = GoogLeNet(in_channels, num_classes).to(device)
    x_out = model(x_in)

    assert x_out.shape == torch.Size([num_samples, num_classes])