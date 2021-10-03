"""Very Deep Convolutional Networks for Large-Scale Image Recognition.

Paper: https://arxiv.org/abs/1409.1556

Created on Sat Oct  2 20:41:48 2021

@author: gonzr
"""

import torch
torch.cuda.empty_cache()
import torch.nn as nn


class VGG(nn.Module):

    def __init__(self):
        super(VGG, self).__init__()

        kernel_size = 3
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=2)
        self.softmax = nn.Softmax()

        self.conv_64 = nn.Conv2d(in_channels=3, out_channels=64,
                                  kernel_size=kernel_size, stride=1, padding=1)
        self.conv_128 = nn.Conv2d(in_channels=64, out_channels=128,
                                  kernel_size=kernel_size, stride=1, padding=1)
        self.conv_256a = nn.Conv2d(in_channels=128, out_channels=256,
                                   kernel_size=kernel_size, stride=1, padding=1)
        self.conv_256b = nn.Conv2d(in_channels=256, out_channels=256,
                                   kernel_size=kernel_size, stride=1, padding=1)
        self.conv_512a = nn.Conv2d(in_channels=256, out_channels=512,
                                   kernel_size=kernel_size, stride=1, padding=1)
        self.conv_512b = nn.Conv2d(in_channels=512, out_channels=512,
                                   kernel_size=kernel_size, stride=1, padding=1)
        self.conv_512c = nn.Conv2d(in_channels=512, out_channels=512,
                                   kernel_size=kernel_size, stride=1, padding=1)
        self.conv_512d = nn.Conv2d(in_channels=512, out_channels=512,
                                   kernel_size=kernel_size, stride=1, padding=1)
        self.fc4096a = nn.Linear(in_features=512, out_features=4096)
        self.fc4096b = nn.Linear(in_features=4096, out_features=4096)
        self.fc1000 = nn.Linear(in_features=4096, out_features=1000)

    def forward(self, x):
        x = self.max_pool(self.relu(self.conv_64(x)))
        x = self.max_pool(self.relu(self.conv_128(x)))
        x = self.relu(self.conv_256a(x))
        x = self.max_pool(self.relu(self.conv_256b(x)))
        x = self.relu(self.conv_512a(x))
        x = self.max_pool(self.relu(self.conv_512b(x)))
        x = self.relu(self.conv_512c(x))
        x = self.max_pool(self.relu(self.conv_512d(x)))
        x = self.relu(self.fc4096a(x))
        x = self.relu(self.fc4096b(x))
        x = self.fc1000(x)
        x = self.softmax(x)
        return x


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
x_in = torch.randn((16, 3, 224, 224), device=device)
model = VGG().to(device)
x_out = model(x_in)









