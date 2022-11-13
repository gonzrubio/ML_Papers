# -*- coding: utf-8 -*-
"""Functionality to visualize the data.

Created on Mon Oct 31 16:03:13 2022

@author: gonzr
"""

import matplotlib.pyplot as plt
import os
import torch

from torchvision.utils import draw_bounding_boxes
from utils.bounding_boxes import yolo_to_voc_bbox, decode_labels
from utils.transforms import InvToTensorNormalize


def plot_batch(
        batch,
        id_class_map,
        id_color_map,
        size=(448, 448),
        fill=True,
        save_dir=None
        ):
    """Plot a collated batch of (image, labels) pairs.

    :param batch: A collated batch of image-label pairs. One of:
        - A batched image tensor and a batched labels tensor.
        - A batched image tensor, a stacked labels tensor and a tensor of batch
        indices mapping each bounding box to its respective image in the batch.
    :type batch: tuple
    :param size: The size to resize the images to, defaults to (448, 448)
    :type size: TYPE, optional
    :param fill: Shade in the bounding boxes, defaults to 'true'
    :type fill: str, optional
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

    inverse_transform = InvToTensorNormalize(size)

    for idx, image in enumerate(images):

        image = inverse_transform(image).to(dtype=torch.uint8)

        if mode == 'train':
            labels_image = decode_labels(labels[idx])
        elif mode == 'eval':
            labels_image = labels[batch_idx == idx]

        boxes = labels_image[:, :-1]
        boxes = yolo_to_voc_bbox(boxes, (image.shape[-2], image.shape[-1]))

        classes = [None] * len(labels_image)
        colors = [None] * len(labels_image)
        for box, class_id in enumerate(labels_image[:, -1]):
            classes[box] = id_class_map[int(class_id.item())]
            colors[box] = id_color_map[int(class_id.item())]

        kwargs = {
            'labels': classes, 'colors': colors, 'fill': fill, 'width': 4,
            }
        drawn_boxes = draw_bounding_boxes(image, boxes, **kwargs)
        drawn_boxes = drawn_boxes.permute(1, 2, 0).numpy()

        plt.imshow(drawn_boxes)
        plt.axis("off")
        plt.show()

        if save_dir:
            plt.imsave(os.path.join(save_dir, f'{idx}.png'), drawn_boxes)
