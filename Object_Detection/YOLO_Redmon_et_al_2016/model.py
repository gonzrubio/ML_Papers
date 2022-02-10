"""You Only Look Once: Unified, Real-Time Object Detection.

Paper: https://arxiv.org/abs/1506.02640

Created on Tue Feb 8 21:38:56 2022

@author: gonzr
"""


import torch
import torch.nn as nn


class YOLO(nn.Module):
    """YOLOv1.

    The network has 24 convolutional layers followed by 2 fully connected
    layers. Alternating 1x1 convolutional layers reduce the feature space from
    preceding layers to reduce computational complexity.
    """

    def __init__(self):
        """Construct the detection network."""
        super(YOLO, self).__init__()
        conv_blocks_config = [
            (7, 64, 2, 3),    # (kernel_size, out_channels, stride, padding)
            "M",              # 2x2-stride and 2x2-kernel_size maxpool
            (3, 192, 1, 1),
            "M",
            (1, 128, 1, 0),
            (3, 256, 1, 1),
            (1, 256, 1, 0),
            (3, 512, 1, 1),
            "M",
            [(1, 256, 1, 0), (3, 512, 1, 1), 4],  # [block1, block2, n_times]
            (1, 512, 1, 0),
            (3, 1024, 1, 1),
            "M",
            [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
            (3, 1024, 1, 1),
            (3, 1024, 2, 1),
            (3, 1024, 1, 1),
            (3, 1024, 1, 1)
            ]

        # add fully connected layers config

        in_channels = 3
        self.conv_blocks = nn.ModuleList([])

        for blk in conv_blocks_config:
            if type(blk) is tuple:
                in_channels = self.__make_conv_block__(in_channels, blk)

            elif type(blk) is str:
                self.conv_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))

            elif type(blk) is list:
                for _i in range(blk[2]):
                    in_channels = self.__make_conv_block__(in_channels, blk[0])
                    in_channels = self.__make_conv_block__(in_channels, blk[1])

    def __make_conv_block__(self, in_channels, block):
        """Append convolutional block and non-linearity to the network.

        :param in_channels: number of input channels
        :type in_channels: int
        :param block: (kernel_size, out_channels, stride, padding)
        :type block: tuple
        :return: out_channels
        :rtype: int
        """
        kernel_size, out_channels, stride, padding = block
        self.conv_blocks.append(
            nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                          padding_mode='reflect',
                          bias=True),
                nn.LeakyReLU(0.1)
            )
        )
        return out_channels

    def forward(self, x):
        """Compute the forward pass.

        :param x: 448x448 rgb input image.
        :type x: TYPE
        :return: 7x7x30 tensor of probabilities.
        :rtype: TYPE
        """
        for block in self.conv_blocks:
            x = block(x)
        # for block in self.fc_layers:
        #     x = block(x)
        return x


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # input batch
    num_samples = 2**0
    in_channels = 3
    H, W = 448, 448

    # network
    model = YOLO().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters for YOLOv1: {total_params:,}")

    # output
    x_out = model(torch.randn((num_samples, in_channels, H, W), device=device))
    print(x_out.shape)
    # assert x_out.shape == torch.Size([num_samples, num_classes])
