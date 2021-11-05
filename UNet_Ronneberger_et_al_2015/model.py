"""U-Net: Convolutional Networks for Biomedical Image Segmentation

Paper: https://arxiv.org/abs/1505.04597

Created on Tue Nov  2 18:27:52 2021

Note: Down sample conv blocks are done with padding and mirroring in the paper.

@author: gonzr
"""


import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64, out_channels=64):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=hidden_channels,
                               kernel_size=3, stride=1, padding=0, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(num_features=hidden_channels)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=1, padding=0, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)

        return x


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        # Downsample conv blocks 5 total, from top to bottom
        n = 5
        self.features = 64
        self.contract = nn.ModuleList()
        self.contract.extend([ConvBlock(in_channels=in_channels,
                                       hidden_channels=self.features,
                                       out_channels=self.features),
                             nn.MaxPool2d(kernel_size=2, stride=2)])

        features = self.features
        for block in range(1, n):
            features *= 2
            self.contract.append(ConvBlock(in_channels=features//2,
                                           hidden_channels=features,
                                           out_channels=features))
            if block < n - 1 :
                self.contract.append(nn.MaxPool2d(kernel_size=2, stride=2))    

        # Expand conv blocks from bottom to top)
        # self.expand = nn.ModuleList()
        # Upsample so it matches the shape of the contraction path
        # read about trsnpose conv
        # self.expand.append()

        # concat skip connections along channels dimension
        # output segmentation map

    def forward(self, x):

        for block in self.contract:
            x = block(x)
            print(x.shape)


        # Copy crop for skip connections:
        return x

if __name__ == '__main__':
    # downsample conv
    # x = torch.randn((1, 1, 572, 572))
    # conv_block = ConvBlock(in_channels=1, hidden_channels=64, out_channels=64)
    # x = conv_block(x)
    # print(x.shape)

    # upsample conv
    # x = torch.randn((1, 256, 200, 200))
    # conv_block = ConvBlock(in_channels=256, hidden_channels=128, out_channels=64)
    # x = conv_block(x)
    # print(x.shape)
    
    # output conv
    # x = torch.randn((1, 128, 392, 392))
    # conv_block = ConvBlock(in_channels=128, hidden_channels=64, out_channels=64)
    # x = conv_block(x)
    # print(x.shape)

    # contracting path
    x = torch.randn((1, 1, 572, 572)) 
    model = UNet()
    x_out = model(x)
    print(x_out.shape)





























