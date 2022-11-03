# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 18:53:53 2022

@author: gonzr
"""

import torch.nn.functional as F

from torchvision import transforms


class Transform(object):
    def __init__(self):
        self.base_transform = transforms.ToTensor()

    def __call__(self, image, labels):
        return self.base_transform(image), labels


class AugmentTransform(Transform):
    def __init__(self):
        super(AugmentTransform, self).__init__()
        self.transforms = [
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]

    def __call__(self, image, labels):
        image, labels = super(AugmentTransform, self).__call__(image, labels)
        for transform in self.transforms:
            image = transform(image)

        return image, labels
