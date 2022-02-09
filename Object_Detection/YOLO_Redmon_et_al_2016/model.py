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
        for block in conv_blocks_config:

            # Refactor this block into a class method (protect it from
            # external use, decorator?), and use again in type(list)
            # add leaky relu
            if type(block) is tuple:
                kernel_size, out_channels, stride, padding = block
                self.conv_blocks.append(nn.Conv2d(in_channels=in_channels,
                                                  out_channels=out_channels,
                                                  kernel_size=kernel_size,
                                                  stride=stride,
                                                  padding=padding,
                                                  padding_mode='reflect',
                                                  bias=False))
                in_channels = out_channels

            elif type(block) is str:
                self.conv_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))

            elif type(block) is list:
                block1, block2, n_times = block
                for _i in range(n_times):
                    kernel_size, out_channels, stride, padding = block1
                    self.conv_blocks.append(nn.Conv2d(in_channels=in_channels,
                                                      out_channels=out_channels,
                                                      kernel_size=kernel_size,
                                                      stride=stride,
                                                      padding=padding,
                                                      padding_mode='reflect',
                                                      bias=False))
                    in_channels = out_channels

                    kernel_size, out_channels, stride, padding = block2
                    self.conv_blocks.append(nn.Conv2d(in_channels=in_channels,
                                                      out_channels=out_channels,
                                                      kernel_size=kernel_size,
                                                      stride=stride,
                                                      padding=padding,
                                                      padding_mode='reflect',
                                                      bias=False))
                    in_channels = out_channels

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
    print(f"Number of parameters for ligand network: {total_params:,}")

    # output
    x_out = model(torch.randn((num_samples, in_channels, H, W), device=device))

    # assert x_out.shape == torch.Size([num_samples, num_classes])
