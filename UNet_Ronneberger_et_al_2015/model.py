"""U-Net: Convolutional Networks for Biomedical Image Segmentation

Paper: https://arxiv.org/abs/1505.04597

Created on Tue Nov  2 18:27:52 2021

Note: Down sample conv blocks are done with padding and mirroring in the paper.

@author: gonzr
"""


import torch
import torch.nn as nn

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()


class ConvBlock(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64, out_channels=64, mirroring=False):
        super(ConvBlock, self).__init__()
        p = 0 if mirroring else 1
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=hidden_channels,
                               kernel_size=3, stride=1, padding=p, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(num_features=hidden_channels)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=1, padding=p, bias=False)
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
    def __init__(self, in_channels=1, out_channels=2, mirroring=False):
        super(UNet, self).__init__()

        # Downsample conv blocks, n total, double the feature size every block
        n = 5
        self.features = 64
        self.contract = nn.ModuleList()
        self.contract.extend([ConvBlock(in_channels=in_channels,
                                       hidden_channels=self.features,
                                       out_channels=self.features,
                                       mirroring=mirroring),
                             nn.MaxPool2d(kernel_size=2, stride=2)])

        features = self.features
        for block in range(1, n):
            features *= 2
            self.contract.append(ConvBlock(in_channels=features//2,
                                           hidden_channels=features,
                                           out_channels=features,
                                           mirroring=mirroring))
            if block < n - 1 :
                self.contract.append(nn.MaxPool2d(kernel_size=2, stride=2))    

        # Expand conv blocks from bottom to top, half features every block
        self.expand = nn.ModuleList()
        for block in range(n - 1):
            features //= 2 
            self.expand.extend([nn.ConvTranspose2d(in_channels=features*2,
                                                  out_channels=features,
                                                  kernel_size=2,
                                                  stride=2,
                                                  padding=0),
                                ConvBlock(in_channels=features*2,
                                          hidden_channels=features,
                                          out_channels=features,
                                          mirroring=mirroring)])

        # output segmentation map
        self.output = nn.Conv2d(in_channels=self.features,
                                out_channels=out_channels,
                                kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        for block in self.contract:
            x = block(x)
            print(x.shape)
            # Copy output for skip connections

        for block_idx, operation in enumerate(self.expand, start=1):
            if block_idx % 2 == 0:
                # concat skip connections along channels dimension
                x = torch.cat((x, x), dim=1)
            x = operation(x)
            print(x.shape)

        x = self.output(x)

        return x

if __name__ == '__main__':

    x = torch.randn((1, 1, 572, 572), device=device) 
    model = UNet(mirroring=True).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params:,}")

    x_out = model(x)
    print(x_out.shape)


