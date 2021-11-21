"""Image-to-Image Translation with Conditional Adversarial Networks.

Paper: https://arxiv.org/abs/1611.07004

Created on Thu Nov 18 17:34:38 2021

@author: gonzr
"""


import torch
import torch.nn as nn


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()


def make_conv(in_size, out_size, batch_norm, leaky):
    """Convolutional blocks of the Discriminator.

    Let Ck denote a Convolution-BtachNorm-ReLU block with k channels.
    All convolutions are 4 x 4 spatial filters with stride 2 and
    downsample by a factor of 2. BatchNorm is not applied to the c64 block.

    After the C512 block, a convolution is applied to map to a 1-d output,
    followed by a Sigmoid function. All ReLUs are leaky with slope of 0.2.
    """
    block = [nn.Conv2d(in_size, out_size,
                       kernel_size=4, stride=2, padding=2,
                       padding_mode="reflect",
                       bias=False if batch_norm else True)]
    if batch_norm:
        block.append(nn.BatchNorm2d(out_size))
    if leaky:
        block.append(nn.LeakyReLU(0.2))
    else:
        block.append(nn.Sigmoid())

    return nn.Sequential(*block)


def init_weights(model, mean=0.0, std=0.02):
    """Initialize weights from a Gaussian distribution."""
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
            nn.init.normal_(module.weight.data, mean=mean, std=std)


# class Generator(nn.Module):
#     def __init__(self, z_dim=100):
#         super().__init__()
#         self.gen = nn.Sequential(
#             self.make_upsample(in_channels=z_dim, out_channels=1024,
#                                 kernel_size=4, stride=1, padding=0),
#             self.make_upsample(in_channels=1024, out_channels=512,
#                                 kernel_size=4, stride=2, padding=1),
#             self.make_upsample(in_channels=512, out_channels=256,
#                                 kernel_size=4, stride=2, padding=1),
#             self.make_upsample(in_channels=256, out_channels=128,
#                                 kernel_size=4, stride=2, padding=1),
#             self.make_upsample(in_channels=128, out_channels=3, bn=False,
#                                 kernel_size=1, stride=1, padding=0),
#             nn.Tanh()
#         )
#         init_weights(mean=0.0, std=0.02)

#     def forward(self, x):
#         return self.gen(x)


class Discriminator(nn.Module):
    """C64-C128-C256-C512 PatchGAN Discriminator architecture."""

    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        channels = [in_channels, 64, 128, 256, 512, 1]
        layers = len(channels)
        channels = zip(channels, channels[1:])

        self.blocks = nn.ModuleList()
        for layer, (input_size, output_size) in enumerate(channels):
            if layer == 0:
                input_size *= 2
                batch_norm = False
                leaky = True
            elif layer < layers - 2:
                batch_norm = True
                leaky = True
            else:
                batch_norm = False
                leaky = False
            self.blocks.append(make_conv(in_size=input_size,
                                         out_size=output_size,
                                         batch_norm=batch_norm,
                                         leaky=leaky))

        init_weights(self, mean=0.0, std=0.02)

    def forward(self, x, y):
        """Output an nxn tensor belonging to patch ij in the input image."""
        x = torch.cat((x, y), dim=1)
        for block in self.blocks:
            x = block(x)
        return x


if __name__ == '__main__':

    batch_size = 16
    channels = 3
    height = 512
    width = 512

    x = torch.randn((batch_size, channels, height, width), device=DEVICE)
    y = torch.randn((batch_size, channels, height, width), device=DEVICE)

    # z = torch.randn((batch_size, 100, 1, 1))
    # generator = Generator()
    # G_z = generator(z)

    # D_z = discriminator(G_z)

    discriminator = Discriminator().to(DEVICE)
    total_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Number of parameters: {total_params:,}")

    D_x = discriminator(x, y)
    print(D_x.shape)

