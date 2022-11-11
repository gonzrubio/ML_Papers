# -*- coding: utf-8 -*-
"""Custom transforms to preprocess, augment and visualize data.

Created on Mon Oct 31 18:53:53 2022

@author: gonzr
"""

import random
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

IMAGENET_NORMALIZE = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
    }


class Augment(object):
    def __init__(self):
        self.transforms = [
                ToTensorNormalize(),
                # RandomTranslate(),
                RandomScale(),
                # RandomJitter(),
                ]

    def __call__(self, image, labels):
        for transform in self.transforms:
            image, labels = transform(image, labels)
        return image, labels


class ToTensorNormalize(object):
    """Convert the PIL Image to a torch image tensor and Normalize."""

    def __init__(self):
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


class InvToTensorNormalize(object):
    pass


class RandomTranslate(object):
    # random horizontal and vertical translations (up to 20% of the image size)
    # translate (tuple, optional) – tuple of maximum absolute fraction for
    # horizontal and vertical translations. For example translate=(a, b), then
    # horizontal shift is randomly sampled in the range
    # -img_width * a < dx < img_width * a and vertical shift is randomly sampled
    # in the range -img_height * b < dy < img_height * b. Will not translate by
    # default.
    def __init__(self, translate=(0.2, 0.2)):
        self.translate_dx, self.translate_dy = translate

    def __call__(self, image, labels):
        # chose randomly in the range -img_width * a < dx < img_width * a
        dx = random.uniform(- self.translate_dx, self.translate_dx)
        dy = random.uniform(- self.translate_dy, self.translate_dy)
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


class RandomScale(object):
    # random scaling (up to 20% of the image size)
    # scale (tuple, optional) – scaling factor interval, e.g (a, b), then scale is
    # randomly sampled from the range a <= scale <= b. Will keep original scale by
    # default.
    def __init__(self, scale=(0.8, 1.2)):
        self.scale = scale

    def __call__(self, img, labels):
        scale = random.uniform(*self.scale)
        img = TF.affine(img, angle=0, shear=0, translate=(0, 0), scale=scale)

        # distance from center of bounding box to center of image
        # image_center = 0.5 * image.shape[-1]
        # cx = cy since its a square image
        image_center = 0.5
        deltas = labels[:, :2] - image_center

        labels[:, :2] = image_center + scale * deltas
        labels[:, 2:4] *= scale

        labels = labels[
            torch.logical_and(
                torch.logical_and(labels[:, 0] >= 0, labels[:, 1] >= 0),
                torch.logical_and(labels[:, 0] <= 1, labels[:, 1] <= 1)
                )
            ]

        # clip width and height of bounding box to the size of the image
        labels[:, 2:4] = torch.clip(labels[:, 2:4], max=1)

        return img, labels


class RandomJitter(object):
    # randomly adjust the exposure and saturation of the image by up to a
    # factor of 1.5 in the HSV color space
    pass
