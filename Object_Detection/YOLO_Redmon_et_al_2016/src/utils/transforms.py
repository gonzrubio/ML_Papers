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
                RandomTranslate(),
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
    """Randomly translate vertically and horizontally the image and the labels.

    The vertical and horizontal shift amounts are sampled independently and
    uniformly from the range -img_shape * a < dx < img_shape * a. Both of the
    translations are at most 20% of the image size.
    """

    def __init__(self, translate=(0.2, 0.2)):
        self.translate_dx, self.translate_dy = translate

    def __call__(self, img, labels):
        """Translate the image and the bounding boxes by a random factor.

        :param img: The original image tensor
        :type img: torch.Tensor
        :param labels: The original ground truth labels
        :type labels: torch.Tensor
        :return: The randomly translated image tensor and bounding boxes
        :rtype: tuple

        """
        dx = random.uniform(- self.translate_dx, self.translate_dx)
        dy = random.uniform(- self.translate_dy, self.translate_dy)
        translate = round(img.shape[1] * dx), round(img.shape[2] * dy)
        img = TF.affine(img, angle=0, shear=0, translate=translate, scale=1)

        # apply translation only to the centers of the bounding boxes
        labels[:, 0] += dx
        labels[:, 1] += dy

        # remove bounding boxes outside of the image
        labels = labels[
            torch.logical_and(
                torch.logical_and(labels[:, 0] >= 0, labels[:, 1] >= 0),
                torch.logical_and(labels[:, 0] <= 1, labels[:, 1] <= 1)
                )
            ]

        return img, labels


class RandomScale(object):
    """Randomly scale the image tensor and the ground truth labels.

    The scale is uniformly sampled from the range a <= scale <= b such that the
    scaling is up to 20% of the image size
    """

    def __init__(self):
        self.scale = (0.8, 1.2)

    def __call__(self, img, labels):
        """Scale the image and the bounding box coordinates by a random factor.

        :param img: The original image tensor
        :type img: torch.Tensor
        :param labels: The original ground truth labels
        :type labels: torch.Tensor
        :return: The randomly scaled image tensor and bounding boxes
        :rtype: tuple

        """
        scale = random.uniform(*self.scale)
        img = TF.affine(img, angle=0, shear=0, translate=(0, 0), scale=scale)

        # get the distances from the centers of the bounding boxs to center of
        # the image and position the centers at the scaled distances
        image_center = 0.5                       # (0.5, 0.5) since normalized
        deltas = labels[:, :2] - image_center    # original distances

        # apply the scaling to the bounding box parameters
        labels[:, :2] = image_center + scale * deltas
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

        return img, labels


class RandomJitter(object):
    # randomly adjust the exposure and saturation of the image by up to a
    # factor of 1.5 in the HSV color space
    pass
