"""Image-to-Image Translation with Conditional Adversarial Networks.

Paper: https://arxiv.org/abs/1611.07004

Created on Thu Nov 18 17:34:38 2021

@author: gonzr
"""


import torch
import torch.nn as nn


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class Discriminator(nn.Module):
    """The 70 x 70 discriminator architecture.

        C64-C128-C256-C512 where Ck denotes a Convolution-BtachNorm-ReLU with
        k channels. All convolutions are 4 x 4 spatial filters applied with
        stride 2 and downsample by a factor of 2. BatchNorm is not applied to
        the c64 layer. After the last layer, a convolution is applied to map
        to a 1-d output, followed by a Sigmoid function. All ReLUs are leaky
        with a slope of 0.2.
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        channels = [64, 128, 256, 512]
        layers = len(channels)
        channels = zip(channels, channels[1:])

        modules = []
        for layer, input_size, output_size in enumerate(channels):
            if layer == 0:
                modules.append(self.make_conv(in_channels=input_size,
                                              out_channels=output_size,
                                              batch_norm=True))
            elif layer < layers:
                modules.append(self.make_conv(in_channels=input_size,
                                              out_channels=output_size,
                                              batch_norm=False))
            else:
                modules.append(nn.Conv2d(in_channels=channels[-1],
                                         out_channels=1,
                                         size=2, kernel=4, stride=2))
                modules.append(nn.Sigmoid())

        self.disc = nn.Sequential(**modules)
        self.init_weights(mean=0.0, std=0.02)

    def make_conv(self, in_channels, out_channels, batch_norm=True):

        layer = [nn.Conv2d(in_channels, out_channels,
                           kernel_size=5, stride=2, padding=2)]
        if batch_norm:
            layer.append(nn.BatchNorm2d(out_channels))
        layer.append(nn.LeakyReLU(0.2))

        return nn.Sequential(*layer)

    def init_weights(self, mean=0.0, std=0.02):
        for module in self.modules():
            if isinstance(module,
                          (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(module.weight.data, mean=mean, std=std)

    def forward(self, x):
        return self.disc(x).view(-1, 1)



class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.gen = nn.Sequential(
            self.make_upsample(in_channels=z_dim, out_channels=1024,
                               kernel_size=4, stride=1, padding=0),
            self.make_upsample(in_channels=1024, out_channels=512,
                               kernel_size=4, stride=2, padding=1),
            self.make_upsample(in_channels=512, out_channels=256,
                               kernel_size=4, stride=2, padding=1),
            self.make_upsample(in_channels=256, out_channels=128,
                               kernel_size=4, stride=2, padding=1),
            self.make_upsample(in_channels=128, out_channels=3, bn=False,
                               kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )
        self.init_weights(mean=0.0, std=0.02)

    def make_upsample(self, in_channels, out_channels, bn=True, **kwargs):

        layer = [nn.ConvTranspose2d(in_channels, out_channels, **kwargs)]
        if bn:
            layer.append(nn.BatchNorm2d(out_channels))
        # layer.append(nn.LeakyReLU(0.2))
        layer.append(nn.ReLU())

        return nn.Sequential(*layer)

    def init_weights(self, mean=0.0, std=0.02):
        for module in self.modules():
            if isinstance(module,
                          (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(module.weight.data, mean=mean, std=std)

    def forward(self, x):
        return self.gen(x)


if __name__ == '__main__':
    z = torch.randn((64, 100, 1, 1))
    generator = Generator()
    G_z = generator(z)

    x = torch.randn((64, 3, 32, 32))
    discriminator = Discriminator()
    D_x = discriminator(x)
    D_z = discriminator(G_z)