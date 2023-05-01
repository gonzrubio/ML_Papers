#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The discriminative and generative adversarial networks.

Created on Sun Apr 23 18:33:05 2023

@author: gonzalo
"""

import torch.nn as nn


class D(nn.Module):
    def __init__(self, input_size, p=0.2):
        super(D, self).__init__()

        self.hidden0 = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(p=p)
            )

        self.hidden1 = nn.Sequential(            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=p)
            )

        self.hidden2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=p)
            )
        
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x


class G(nn.Module):
    def __init__(self, input_size, output_size):
        super(G, self).__init__()

        self.hidden0 = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU()
            )

        self.hidden1 = nn.Sequential(            
            nn.Linear(128, 256),
            nn.ReLU()
            )

        self.hidden2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU()
            )

        self.out = nn.Sequential(
            nn.Linear(512, output_size),
            nn.Tanh()  # -1, 1 since the real images are standardized
            )


    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
