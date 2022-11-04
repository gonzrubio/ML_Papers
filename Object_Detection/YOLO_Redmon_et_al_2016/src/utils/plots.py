# -*- coding: utf-8 -*-
"""Functionality to visualize the data.

Created on Mon Oct 31 16:03:13 2022

@author: gonzr
"""

import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F

from .bounding_boxes import yolo_to_voc_bbox, decode_labels
from torchvision.utils import draw_bounding_boxes


# https://gist.github.com/anujonthemove/d6d84be473e27057f8d95b89bac50cf9
CLASS_ID_MAP = {
    'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5,
    'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
    'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16,
    'sofa': 17, 'train': 18, 'tvmonitor': 19
    }

ID_CLASS_MAP = {
    0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle', 5: 'bus',
    6: 'car', 7: 'cat', 8: 'chair', 9: 'cow', 10: 'diningtable', 11: 'dog',
    12: 'horse', 13: 'motorbike', 14: 'person', 15: 'pottedplant', 16: 'sheep',
    17: 'sofa', 18: 'train', 19: 'tvmonitor'
    }

# Color map for bounding boxes of detected objects from
# https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
ID_COLOR_MAP = {
    0: '#e6194b', 1: '#3cb44b', 2: '#ffe119', 3: '#0082c8', 4: '#f58231',
    5: '#911eb4', 6: '#46f0f0', 7: '#f032e6', 8: '#d2f53c', 9: '#fabebe',
    10: '#008080', 11: '#000080', 12: '#aa6e28', 13: '#fffac8', 14: '#800000',
    15: '#aaffc3', 16: '#808000', 17: '#ffd8b1', 18: '#e6beff', 19: '#808080'
    }


def plot_batch(batch, size=(1024, 1024), save_dir=None):
    """Plot a collated batch of (image, labels) pairs.

    :param batch: A collated batch of image-label pairs. One of:
        - A batched image tensor and a batched labels tensor.
        - A batched image tensor, a stacked labels tensor and a tensor of batch
        indices mapping each bounding box to its respective image in the batch.
    :type batch: tuple
    :param size: The size to resize the images to, defaults to (1024, 1024)
    :type resize_size: TYPE, optional
    :param save_dir: If specified, the directory to save the plotted images to,
    defaults to None
    :type save_dir: str, optional

    """
    if batch[-1].dim() == 4:
        mode = 'train'
        (images, labels) = batch
    elif batch[-1].dim() == 1:
        mode = 'eval'
        (images, labels, batch_idx) = batch

    for idx, image in enumerate(images):

        image = F.interpolate(image.unsqueeze(0), size=size).squeeze(0)
        image = (image * 255).to(dtype=torch.uint8)

        if mode == 'train':
            labels_image = decode_labels(labels[idx])
        elif mode == 'eval':
            labels_image = labels[batch_idx == idx]

        boxes = labels_image[:, :-1]
        boxes = yolo_to_voc_bbox(boxes, (image.shape[-2], image.shape[-1]))

        classes = [None] * len(labels_image)
        colors = [None] * len(labels_image)
        for box, class_id in enumerate(labels_image[:, -1]):
            classes[box] = ID_CLASS_MAP[int(class_id.item())]
            colors[box] = ID_COLOR_MAP[int(class_id.item())]

        kwargs = {'labels': classes, 'colors': colors}
        drawn_boxes = draw_bounding_boxes(image, boxes, **kwargs)
        drawn_boxes = drawn_boxes.permute(1, 2, 0).numpy()

        plt.imshow(drawn_boxes)
        plt.axis("off")
        plt.show()

        if save_dir:
            plt.imsave(os.path.join(save_dir, f'{idx}.png'), drawn_boxes)
