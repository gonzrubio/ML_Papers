# -*- coding: utf-8 -*-
"""Custom transforms to preprocess, augment and visualize data.

Created on Mon Oct 31 18:53:53 2022

@author: gonzr
"""

import random
import torch
import torchvision.transforms.functional as TF

IMAGENET_NORMALIZE = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
    }


class ToTensorNormalize(object):
    """Convert the PIL Image to a torch image tensor and Normalize."""

    def __init__(self):
        super(ToTensorNormalize, self).__init__()
        self.mean = IMAGENET_NORMALIZE['mean']
        self.std = IMAGENET_NORMALIZE['std']

    def __call__(self, image, labels):
        """Convert the image to a tensor and normalize using ImageNet.

        :param image: The raw image
        :type image: PIL.Image
        :param labels: The labels for the objects in the image. Left unchanged.
        :type labels: torch.Tensor
        :return: The normalized tensor image and original labels
        :rtype: tuple

        """
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=self.mean, std=self.std)
        return image, labels


class Augment(ToTensorNormalize):
    # random translation and scaling up to 20% in all directions
    # translate (tuple, optional) – tuple of maximum absolute fraction for
    # horizontal and vertical translations. For example translate=(a, b), then
    # horizontal shift is randomly sampled in the range
    # -img_width * a < dx < img_width * a and vertical shift is randomly sampled
    # in the range -img_height * b < dy < img_height * b. Will not translate by
    # default.
    # scale (tuple, optional) – scaling factor interval, e.g (a, b), then scale is
    # randomly sampled from the range a <= scale <= b. Will keep original scale by
    # default.
    def __init__(
            self,
            translate=(0.2, 0.2),
            scale=(0.8, 1.2)
            ):
        super(Augment, self).__init__()
        self.translate = translate
        self.scale = scale

        # can group sequential transforms using Compose()
        # self.transforms = [
        #     ]

        # randomly adjust the exposure and saturation of the image by up to a
        # factor of 1.5 in the HSV color space

    def __call__(self, image, labels):
        image, labels = super(Augment, self).__call__(image, labels)
        image, labels = self.random_affine(image, labels)
        return image, labels

    def random_affine(self, image, labels):
        # chose randomly in the range -img_width * a < dx < img_width * a
        dx = random.uniform(- self.translate[0], self.translate[0])
        dy = random.uniform(- self.translate[1], self.translate[1])
        translate = round(image.shape[1] * dx), round(image.shape[2] * dy)
        # scale = random.uniform(*self.scale)
        # dx = 0
        # dy = 0
        scale = 1

        image = TF.affine(
            image, angle=0, shear=0, translate=translate, scale=scale
            )

        # apply transform to labels
        labels[:, 0] += dx
        labels[:, 1] += dy
        labels[:, 2:4] *= scale

        # remove bounding boxes outside of the image
        labels = labels[
            torch.logical_and(
                torch.logical_and(labels[:, 0] >= 0, labels[:, 1] >= 0),
                torch.logical_and(labels[:, 0] <= 1, labels[:, 1] <= 1)
                )
            ]

        # clip width and height of bounding box to the size of the image
        labels[:, 2:4] = torch.clip(labels[:, 2:4], max=1)

        return image, labels





















