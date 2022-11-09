# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 18:53:53 2022

@author: gonzr
"""

from torchvision import transforms

IMAGENET_NORMALIZE = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
    }


class ToTensor(object):
    def __init__(self):
        super(ToTensor, self).__init__()
        # wrapper for torchvision.transforms.ToTensor() to convert an image
        # in ... to .... , expand on this
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image, labels):
        return self.to_tensor(image), labels


class Augment(ToTensor):
    def __init__(self):
        super(Augment, self).__init__()
        # normalize the image tensor values with the mean and standard
        # deviation of ImageNet
        # random scaling and translations up to 20%
        # randomly adjust the exposure and saturation of the image by up to a
        # factor of 1.5 in the HSV color space
        # self.transforms = [
        #     ]
        pass

    def __call__(self, image, labels):
        image, labels = super(Augment, self).__call__(image, labels)
        # for transform in self.transforms:
        #     image = transform(image)

        return image, labels
