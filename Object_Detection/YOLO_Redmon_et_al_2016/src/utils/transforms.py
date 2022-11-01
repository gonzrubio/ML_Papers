# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 18:53:53 2022

@author: gonzr
"""

import torch.nn.functional as F

from torchvision import transforms


class Transform(object):
    def __init__(self):
        self.transforms = [
            transforms.Resize((448, 448)),
            transforms.ToTensor()
            ]

    def __call__(self, image, labels):
        for transform in self.transforms:
            image = transform(image)

        return image, labels


class AugmentTransform(Transform):
    def __init__(self):
        super(AugmentTransform, self).__init__()
        self.transforms = [
            transforms.Resize((448, 448)),
            transforms.ToTensor()
            ]

    def __call__(self, image, labels):
        image, labels = self(image, labels)
        for transform in self.transforms:
            image = transform(image)

        return image, labels
