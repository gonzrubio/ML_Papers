#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:17:47 2023

@author: gonzalo
"""

import torch
import torch.nn as nn


class Resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resblock, self).__init__()

        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU()
            )

    def forward(self, x):
        return self.skip(x) + self.conv_block(x)


class StyleNet(nn.Module):
    def __init__(self):
        super(StyleNet, self).__init__()

        rb_channels = 16
        self.scales = 4

        self.rgb2rb = nn.Conv2d(
            in_channels=3, out_channels=rb_channels, kernel_size=1
            )

        # encoder blocks
        self.encoder_blocks = nn.ModuleList()

        for i in range(self.scales):
            in_ch = rb_channels if i == 0 else rb_channels * (2 ** (i - 1))
            out_ch = rb_channels * (2 ** i)
            self.encoder_blocks.append(Resblock(in_ch, out_ch))

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()

        for i in range(self.scales - 1, 0, -1):
            in_ch = rb_channels * (2 ** i) + rb_channels * (2 ** (i - 1))
            out_ch = rb_channels * (2 ** (i - 1))
            self.decoder_blocks.append(Resblock(in_ch, out_ch))

        self.head = nn.Sequential(
            Resblock(out_ch + rb_channels, rb_channels),
            nn.Conv2d(
                in_channels=rb_channels,
                out_channels=3,
                kernel_size=1),
            nn.Sigmoid()            
            )
            
    def forward(self, x):
        

        x = self.rgb2rb(x)
        skip_connections = [x]

        # Encoder
        for i in range(self.scales):
            x = self.encoder_blocks[i](x)
            if i < self.scales - 1:
                skip_connections.append(x)
                # TODO stridded conv filter 4 step 3 pad 1

        # Decoder
        for i in range(self.scales - 1, 0, -1):
            # x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
            # TODO bilinear upsample
            x = torch.cat([x, skip_connections[i]], dim=0)
            x = self.decoder_blocks[self.scales - i - 1](x)

        return self.head(torch.cat([x, skip_connections[i]], dim=0))


if __name__ == '__main__':
    # TODO compare authors stylenet.py file
    # TODO add dosctrings
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = StyleNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters for StyleNet: {total_params:,}")

    x_in = torch.randn((3, 512, 512), device=device, dtype=torch.float32)
    x_out = model(x_in)

    assert x_out.shape == x_in.shape
    assert (x_out.min() > 0) and (x_out.max() < 1)
