"""Very Deep Convolutional Networks for Large-Scale Image Recognition.

Paper: https://arxiv.org/abs/1409.1556

Created on Sat Oct  2 20:41:48 2021

@author: gonzr
"""

import torch
import torch.nn as nn


class VGG(nn.Module):
    """Flexible VGG network."""

    def __init__(self, in_channels, conv_blocks, fc_layers, out_channels):
        """Build custom convulutional blocks and fully connected layers."""
        super(VGG, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_blocks = self.make_conv_blocks(conv_blocks)
        self.fc_layers = self.make_fc_layers(fc_layers)

    def make_conv_blocks(self, specs):
        """Make the custom convolutional blocks."""
        conv = []
        krnl = (3, 3)
        strd = (1, 1)
        pdd = (1, 1)
        strd_pool = (2, 2)
        krnl_pool = (2, 2)
        in_channels = self.in_channels

        for spec in specs:

            if isinstance(spec, int):
                conv += [nn.Conv2d(in_channels=in_channels, out_channels=spec,
                                   kernel_size=krnl, stride=strd, padding=pdd),
                         nn.BatchNorm2d(spec),
                         nn.ReLU()]

            elif spec == 'M':
                conv += [nn.MaxPool2d(kernel_size=krnl_pool, stride=strd_pool)]

            in_channels = spec

        return nn.Sequential(*conv)

    def make_fc_layers(self, specs):
        pass

    def forward(self, x):
        return x


if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    conv = [64, 64, 'M',
            128, 128, 'M',
            256, 256, 256, 'M',
            512, 512, 512, 'M',
            512, 512, 512, 'M']

    fc = [4096, 4096, 1000]

    num_samples = 2**4
    in_channels = 3
    size = 224
    out_channels = fc[-1]

    x_in = torch.randn((num_samples, in_channels, size, size), device=device)
    model = VGG(in_channels, conv, fc, out_channels).to(device)
    x_out = model(x_in)

    assert x_out.shape == [0, 1]
