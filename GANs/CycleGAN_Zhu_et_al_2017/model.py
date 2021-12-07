"""Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.

Paper: https://arxiv.org/abs/1703.10593
Data: https://www.kaggle.com/defileroff/comic-faces-paired-synthetic-v2

Created on Mon Dec  6 17:13:56 2021

@author: gonzr
"""


import torch
import torch.nn as nn


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()


def make_conv(in_size, out_size, encode, instance_norm, activation, drop_out):
    """Convolutional blocks of the Generator and the Discriminator.

    :param in_size: number of input filters
    :type in_size: int
    :param out_size: number of output filters
    :type out_size: int
    :param encode: apply convolution to downsaple
    :type encode: bool
    :param instance_norm: apply instance norm
    :type instance_norm: bool
    :param activation: activation function
    :type activation: str
    :param drop_out: apply 0.5 dropout
    :type drop_out: bool
    :return: the convolutional block
    :rtype: nn.Sequential

    Let Ck denote a Convolution-InstanceNorm-ReLU block with k filters.
    CDk denotes a Convolution-BtachNorm-Dropout-ReLU block with 50% dropout.
    All convolutions are 4 x 4 spatial filters with stride 2. Convolutions in
    the encoder and discriminator downsample by a factor of 2, whereas in the
    decoder they upsample by a factor of 2.
    """
    block = [nn.Conv2d(in_size, out_size,
                       kernel_size=4, stride=2, padding=1,
                       padding_mode="reflect",
                       bias=False if instance_norm else True)
             if encode else
             nn.ConvTranspose2d(in_size, out_size,
                                kernel_size=4, stride=2, padding=1,
                                bias=False if instance_norm else True)]

    if instance_norm:
        block.append(nn.InstanceNorm2d(out_size))
    if activation == "leaky":
        block.append(nn.LeakyReLU(0.2))
    elif activation == "sigmoid":
        block.append(nn.Sigmoid())
    elif activation == "tanh":
        block.append(nn.Tanh())
    elif activation == "relu":
        block.append(nn.ReLU())
    if drop_out:
        block.append(nn.Dropout(0.5))

    return nn.Sequential(*block)


class Discriminator(nn.Module):
    """C64-C128-C256-C512 63x63 PatchGAN.

    After the C512 block, a convolution is applied to map to a 1-d output,
    followed by a Sigmoid function. InstanceNorm is not applied to the first
    c64 layer. All ReLUs are leaky with slope of 0.2.
    """

    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        channels = [in_channels, 64, 128, 256, 512, 1]
        layers = len(channels)
        channels = zip(channels, channels[1:])

        self.blocks = nn.ModuleList()
        for layer, (input_size, output_size) in enumerate(channels):
            instance_norm = False if layer == 0 else True
            activation = "leaky" if layer < layers - 2 else "sigmoid"
            self.blocks.append(make_conv(in_size=input_size,
                                         out_size=output_size,
                                         encode=True,
                                         instance_norm=instance_norm,
                                         activation=activation,
                                         drop_out=False))

    def forward(self, x):
        """Return a nxn tensor of patch probabilities."""
        for block in self.blocks:
            x = block(x)
        return x


if __name__ == '__main__':

    batch_size = 1
    channels = 3
    height = 64
    width = 64

    x = torch.randn((batch_size, channels, height, width), device=DEVICE)
    # y = torch.randn((batch_size, channels, height, width), device=DEVICE)

    # generator = Generator().to(DEVICE)
    # total_params = sum(p.numel() for p in generator.parameters())
    # print(f"Number of parameters in Generator: {total_params:,}")

    # G_z = generator(x, z)
    # print(G_z.shape)
    Discriminator()
    discriminator = Discriminator().to(DEVICE)
    total_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Number of parameters in Discriminator: {total_params:,}")

    D_x = discriminator(x)
    print(D_x.shape)
