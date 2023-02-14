"""Gradient Based Learning Applied to Document Recognition.

Paper: http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf

Created on Fri Oct  1 13:54:54 2021

@author: gonzo
"""


import torch
import torch.nn as nn


class LeNet(nn.Module):
    """LeNet implementation."""

    def __init__(self):
        """Lenet architecture.

        1x32x32 -> ((5x5), s=1, p=0 -> avg pool s=2, p=0 -> (5x5), s=1,p=0) x2
        -> Conv 5x5 to 120 channels -> linear 84 -> linear 10
        """
        super(LeNet, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,
                               kernel_size=(5, 5), stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,
                               kernel_size=(5, 5), stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120,
                               kernel_size=(5, 5), stride=1, padding=0)
        self.linear1 = nn.Linear(in_features=120, out_features=84)
        self.linear2 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        """Forward method."""
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


x_in = torch.randn((64, 1, 32, 32))
model = LeNet()
x_out = model(x_in)
print(x_out.shape)
