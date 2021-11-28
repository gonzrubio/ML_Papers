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
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.batch_norm(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, skip=None):
        super(ResBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=1, padding='same')
        self.conv2 = ConvBlock(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, stride=stride, padding=1)
        self.conv3 = ConvBlock(in_channels=out_channels, out_channels=4*out_channels,
                               kernel_size=1, padding='same')
        self.skip = skip
        self.relu = nn.ReLU()

    def forward(self, x):

        skip = self.skip(x) if self.skip else x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return self.relu(x + skip)


class ResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1000, num_layers=[3, 4, 6, 3]):
        super(ResNet, self).__init__()

        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=64,
                               kernel_size=7, stride=2, padding=3)
        self.conv2_x = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self.make_res_block(in_channels=64, out_channels=64,
                                num_blocks=num_layers[0], stride=1)
            )
        self.conv3_x = self.make_res_block(in_channels=256, out_channels=128,
                                           num_blocks=num_layers[1], stride=2)
        self.conv4_x = self.make_res_block(in_channels=512, out_channels=256,
                                           num_blocks=num_layers[2], stride=2)
        self.conv5_x = self.make_res_block(in_channels=1024, out_channels=512,
                                           num_blocks=num_layers[3], stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(in_features=2048, out_features=out_channels, bias=False)

    def make_res_block(self, in_channels, out_channels, num_blocks, stride):

        skip = ConvBlock(in_channels=in_channels, out_channels=4*out_channels,
                         kernel_size=1, stride=stride, padding=0)

        res_block = [ResBlock(in_channels=in_channels, out_channels=out_channels,
                              stride=stride, skip=skip)]

        in_channels = 4*out_channels
        for block in range(1, num_blocks):
            res_block += [ResBlock(in_channels=in_channels, out_channels=out_channels)]

        return nn.Sequential(*res_block)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


def ResNet50(in_channels=3, out_channels=1000):
    return ResNet(in_channels, out_channels, num_layers=[3, 4, 6, 3])


def ResNet101(in_channels=3, out_channels=1000):
    return ResNet(in_channels, out_channels, num_layers=[3, 4, 23, 3])


def ResNet152(in_channels=3, out_channels=1000):
    return ResNet(in_channels, out_channels, num_layers=[3, 8, 36, 3])


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    num_samples = 2**3
    in_channels = 3
    size = 224
    num_classes = 1000
    x_in = torch.randn((num_samples, in_channels, size, size), device=device)

    model = ResNet50(in_channels, num_classes).to(device)
    x_out = model(x_in)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters for ResNet50: {total_params:,}")

    assert x_out.shape == torch.Size([num_samples, num_classes])

    model = ResNet101(in_channels, num_classes).to(device)
    x_out = model(x_in)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters for ResNet101: {total_params:,}")

    assert x_out.shape == torch.Size([num_samples, num_classes])

    model = ResNet152(in_channels, num_classes).to(device)
    x_out = model(x_in)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters for ResNet152: {total_params:,}")

    assert x_out.shape == torch.Size([num_samples, num_classes])
