# -*- coding: utf-8 -*-
"""Common functionality for bounding boxes.

Created on Fri Oct 28 13:22:59 2022

@author: gonzr
"""

import torch

def encode_labels(labels, S=7):
    # encode the target labels as an S x S x len([cx, cy, w, h, class]) tensor
    # gts [[xc, yc, w, h, class_id]_i, ..., [xc, yc, w, h, class_id]_n]
    # :param S: Number of grid cells to split the image for each direction,
    # defaults to 7
    # :type S: int, optional
    # :param B: Number of predicted bounding boxes per gird cell,
    # defaults to 2
    # :type B: int, optional
    # :param C: Number of class labels, defaults to 20
    # :type C: int, optional

    # divide the image into an S x S grid and assign an object to a grid cell
    # if the center of an object falls into that grid cell
    # the center coordinates of the object are relative to the grid cell and
    # the width and height are relative to the whole image
    # grid cells containing an object also contain the class of the object
    encoded_labels = torch.zeros((S, S, 5), dtype=labels.dtype)

    # if num_obj > 1:
    x, y = S * labels[:, 0], S * labels[:, 1]
    row = torch.floor(x).to(dtype=torch.long)
    col = torch.floor(y).to(dtype=torch.long)
    xc, yc = x - row, y - col
    labels[:, 0], labels[:, 1] = xc, yc
    encoded_labels[row, col, :] = labels

    # if num_obj > 1:
    #     x, y = S * labels[:, 0], S * labels[:, 1]
    #     row = torch.floor(x).to(dtype=torch.long)
    #     col = torch.floor(y).to(dtype=torch.long)
    #     xc, yc = x - row, y - col
    #     labels[:, 0], labels[:, 1] = xc, yc
    #     encoded_labels[row, col, :] = labels
    # else:
    #     x, y = S * labels[0], S * labels[1]
    #     row, col = int(x), int(y)
    #     xc, yc = x - row, y - col
    #     labels[0], labels[1] = xc, yc
    #     encoded_labels[row, col, :] = labels
    return encoded_labels


def yolo_to_voc_bbox(bboxes, im_shape):
    """Convert the bounding boxes to pixel coordinates.

    :param bboxes: The YOLO encoded coordinates for the bouding boxes
    :type labels: torch.Tensor
    :param im_shape: Height and Width of the image
    :type im_shape: tuple of ints
    :return: The pixel coordinates for the bouding boxes
    :rtype: torch.Tensor

    """
    bboxes[:, 0::2] *= im_shape[0]      # scale x-coordinates
    bboxes[:, 1::2] *= im_shape[1]      # scale y-coordinates
    bboxes[:, 0] -= 0.5 * bboxes[:, 2]  # x1
    bboxes[:, 1] -= 0.5 * bboxes[:, 3]  # y1
    bboxes[:, 2] += bboxes[:, 0]        # x2
    bboxes[:, 3] += bboxes[:, 1]        # y2

    return bboxes


def iou(bbox_pred, bbox_true):
    """Compute the intersection over union.

    :param bbox_pred: Predicted bounding boxes (num_boxes, 4)
    :type bbox_pred: torch.Tensor
    :param bbox_true: True bounding boxes (num_boxes, 4)
    :type bbox_true: torch.Tensor
    :return: The intersection over union for all pairs of boxes
    :rtype: torch.Tensor

    """
    # define the rectangles by their top-left and bottom-right coordinates
    box1_x1 = bbox_pred[..., 0:1] - bbox_pred[..., 2:3] / 2
    box1_y1 = bbox_pred[..., 1:2] - bbox_pred[..., 3:4] / 2
    box1_x2 = bbox_pred[..., 0:1] + bbox_pred[..., 2:3] / 2
    box1_y2 = bbox_pred[..., 1:2] + bbox_pred[..., 3:4] / 2

    box2_x1 = bbox_true[..., 0:1] - bbox_true[..., 2:3] / 2
    box2_y1 = bbox_true[..., 1:2] - bbox_true[..., 3:4] / 2
    box2_x2 = bbox_true[..., 0:1] + bbox_true[..., 2:3] / 2
    box2_y2 = bbox_true[..., 1:2] + bbox_true[..., 3:4] / 2

    # intersection width as overlap in x-axis
    w = (torch.min(box1_x2, box2_x2) - torch.max(box1_x1, box2_x1)).clamp(0)
    # intersection height as overlap in y-axis
    h = (torch.min(box1_y2, box2_y2) - torch.max(box1_y1, box2_y1)).clamp(0)
    intersection = w * h

    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union = box1_area + box2_area - intersection

    return intersection / union
