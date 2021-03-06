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


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes, size):
        super(InceptionAux, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.conv = ConvBlock(in_channels=in_channels, out_channels=32,
                              kernel_size=1, padding='same')
        self.linear1 = nn.Linear(in_features=32*size**2, out_features=512)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.4)
        self.linear2 = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.drop_out(self.relu(self.linear1(x)))
        x = self.linear2(x)
        return x


class GoogLeNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1000, aux=False):
        super(GoogLeNet, self).__init__()

        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=64,
                               kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Sequential(
            ConvBlock(in_channels=64, out_channels=64,
                      kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=64, out_channels=192,
                      kernel_size=3, padding='same'))
        self.inception_3a = InceptionBLock(c_in=192, out_1=64,
                                           red_3=96, out_3=128,
                                           red_5=16, out_5=32,
                                           out_1p=32)
        self.inception_3b = InceptionBLock(c_in=256, out_1=128,
                                           red_3=128, out_3=192,
                                           red_5=32, out_5=96,
                                           out_1p=64)
        self.inception_4a = InceptionBLock(c_in=480, out_1=192,
                                           red_3=96, out_3=208,
                                           red_5=16, out_5=48,
                                           out_1p=64)
        self.inception_4b = InceptionBLock(c_in=512, out_1=160,
                                           red_3=112, out_3=224,
                                           red_5=24, out_5=64,
                                           out_1p=64)

        self.inception_4c = InceptionBLock(c_in=512, out_1=128,
                                           red_3=128, out_3=256,
                                           red_5=24, out_5=64,
                                           out_1p=64)
        self.inception_4d = InceptionBLock(c_in=512, out_1=112,
                                           red_3=144, out_3=288,
                                           red_5=32, out_5=64,
                                           out_1p=64)
        self.inception_4e = InceptionBLock(c_in=528, out_1=256,
                                           red_3=160, out_3=320,
                                           red_5=32, out_5=128,
                                           out_1p=128)
        self.inception_5a = InceptionBLock(c_in=832, out_1=256,
                                           red_3=160, out_3=320,
                                           red_5=32, out_5=128,
                                           out_1p=128)
        self.inception_5b = InceptionBLock(c_in=832, out_1=384,
                                           red_3=192, out_3=384,
                                           red_5=48, out_5=128,
                                           out_1p=128)
        self.linear = nn.Linear(in_features=1024, out_features=out_channels)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.drop_out = nn.Dropout(p=0.4)
        self.aux = aux
        if self.aux:
            self.aux1 = InceptionAux(in_channels=480,
                                     num_classes=num_classes,
                                     size=14)
            self.aux2 = InceptionAux(in_channels=832,
                                     num_classes=num_classes,
                                     size=7)

    def forward(self, x):
        x = self.max_pool(self.conv1(x))
        x = self.max_pool(self.conv2(x))
        x = self.max_pool(self.inception_3b(self.inception_3a(x)))
        # Auxiliary Softmax classifier 1
        if self.aux:
            aux1 = self.aux1(x)
        x = self.inception_4c((self.inception_4b(self.inception_4a(x))))
        x = self.max_pool(self.inception_4e(self.inception_4d(x)))
        # Auxiliary Softmax classifier 2
        if self.aux:
            aux2 = self.aux2(x)
        x = self.avg_pool(self.inception_5b(self.inception_5a(x)))
        x = x.reshape(x.shape[0], -1)
        x = self.drop_out(x)
        x = self.linear(x)
        return aux1, aux2, x if self.aux else x


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    num_samples = 2**4
    in_channels = 3
    size = 224
    num_classes = 1000

    x_in = torch.randn((num_samples, in_channels, size, size), device=device)
    model = GoogLeNet(in_channels, num_classes, aux=True).to(device)
    x_out = model(x_in)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters for ligand network: {total_params:,}")
    assert x_out.shape == torch.Size([num_samples, num_classes])
