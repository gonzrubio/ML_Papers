"""Image-to-Image Translation with Conditional Adversarial Networks.

Paper: https://arxiv.org/abs/1611.07004

Created on Thu Nov 18 17:34:38 2021

@author: gonzr
"""


import torch
import torch.nn as nn


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()


def make_conv(in_size, out_size, encode, batch_norm, activation, drop_out):
    """Convolutional blocks of the Generator and the Discriminator.

    Let Ck denote a Convolution-BtachNorm-ReLU block with k filters.
    CDk denotes a Convolution-BtachNorm-Dropout-ReLU block with 50% dropout.
    All convolutions are 4 x 4 spatial filters with stride 2. Convolutions in
    the encoder and discriminator downsample by a factor of 2, whereas in the
    decoder they upsample by a factor of 2.
    """
    block = [nn.Conv2d(in_size, out_size,
                       kernel_size=4, stride=2, padding=1,
                       padding_mode="reflect",
                       bias=False if batch_norm else True)
             if encode else
             nn.ConvTranspose2d(in_size, out_size,
                                kernel_size=4, stride=2, padding=1,
                                bias=False if batch_norm else True)]

    if batch_norm:
        block.append(nn.BatchNorm2d(out_size))
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


def init_weights(model, mean=0.0, std=0.02):
    """Initialize weights from a Gaussian distribution."""
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
            nn.init.normal_(module.weight.data, mean=mean, std=std)


class Generator(nn.Module):
    """UNet Generator architecture.

    encoder:
        C64-C128-C256-C512-C512-C512-C512-C512
    decoder:
        CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128

    After the C128 block in the decoder, a convolution is applied to map to the
    number of output channels, followed by a Tanh function. BatchNorm is not
    applied to the C64 block in the encoder. All ReLUs in the econder are
    leaky with slope 0.2, while ReLUs in the decoder are not leaky.
    """

    def __init__(self, in_channels=3, out_channels=3):
        super(Generator, self).__init__()

        encoder = [in_channels, 64, 128, 256, 512, 512, 512, 512, 512]
        encoder = zip(encoder, encoder[1:])

        self.encoder = nn.ModuleList()
        for idx, (input_size, output_size) in enumerate(encoder):
            if idx == 0:
                input_size *= 2
                batch_norm = False
            else:
                batch_norm = True
            self.encoder.append(make_conv(in_size=input_size,
                                          out_size=output_size,
                                          encode=True,
                                          batch_norm=batch_norm,
                                          activation="leaky",
                                          drop_out=False))

        decoder = [512, 1024, 1024, 1024, 1024, 512, 256, 128, out_channels]
        layers_decoder = len(decoder)
        decoder = zip(decoder, decoder[1:])

        self.decoder = nn.ModuleList()
        for idx, (input_size, output_size) in enumerate(decoder):
            if idx < layers_decoder - 2:
                batch_norm = True
                activation = "relu"
                output_size //= 2
            else:
                batch_norm = False
                activation = "tanh"
            self.decoder.append(make_conv(in_size=input_size,
                                          out_size=output_size,
                                          encode=False,
                                          batch_norm=batch_norm,
                                          activation=activation,
                                          drop_out=True if idx < 3 else False))

        init_weights(self, mean=0.0, std=0.02)

    def forward(self, x, z):
        """Generate a translation of x conditioned on the noise z."""
        x = torch.cat((x, z), dim=1)
        skip = [None]*len(self.encoder)

        for idx, block in zip(range(len(skip)-1, -1, -1), self.encoder):
            x = block(x)
            skip[idx] = x

        for idx, block in enumerate(self.decoder):
            if idx > 0:
                x = torch.cat((x, skip[idx]), dim=1)
            x = block(x)

        return x


class Discriminator(nn.Module):
    """C64-C128-C256-C512 PatchGAN Discriminator architecture.

    After the C512 block, a convolution is applied to map to a 1-d output,
    followed by a Sigmoid function. BatchNorm is not applied to the c64 block.
    All ReLUs are leaky with slope of 0.2.
    """

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
                activation = "leaky"
            elif layer < layers - 2:
                batch_norm = True
                activation = "leaky"
            else:
                batch_norm = False
                activation = "sigmoid"
            self.blocks.append(make_conv(in_size=input_size,
                                         out_size=output_size,
                                         encode=True,
                                         batch_norm=batch_norm,
                                         activation=activation,
                                         drop_out=False))

        init_weights(self, mean=0.0, std=0.02)

    def forward(self, x, y):
        """Return a nxn tensor of patch probabilities."""
        x = torch.cat((x, y), dim=1)
        for block in self.blocks:
            x = block(x)
        return x


if __name__ == '__main__':

    batch_size = 8
    channels = 3
    height = 256
    width = 256

    x = torch.randn((batch_size, channels, height, width), device=DEVICE)
    y = torch.randn((batch_size, channels, height, width), device=DEVICE)
    z = torch.randn((batch_size, channels, height, width), device=DEVICE)

    generator = Generator().to(DEVICE)
    total_params = sum(p.numel() for p in generator.parameters())
    print(f"Number of parameters in Generator: {total_params:,}")

    G_z = generator(x, z)
    print(G_z.shape)

    discriminator = Discriminator().to(DEVICE)
    total_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Number of parameters in Discriminator: {total_params:,}")

    D_x = discriminator(x, y)
    print(D_x.shape)
