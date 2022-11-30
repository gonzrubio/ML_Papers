# -*- coding: utf-8 -*-
"""Common functionality for bounding boxes.

Created on Fri Oct 28 13:22:59 2022

@author: gonzr
"""

import torch

from torchvision.ops import nms


def detect_objects(pred_labels):
    """Convert the predicted tensor to a list of objects.

    Performs non-maximum supression on the predicted labels above a probability
    score and sorts the kept elements in decreasing order of score.

    :param pred_labels: The predicted output tensor
    :type pred_labels: torch.Tensor
    :return: The filtered predicted tensor as a list of objects
    :rtype: torch.Tensor

    """
    pred_labels = decode_predicted_labels(pred_labels)
    pred_labels = pred_labels[pred_labels[:, 0] > 0.2]
    idx_keep = nms(
        boxes=yolo_to_voc_bbox(pred_labels[:, 1:5], (1, 1)),
        scores=pred_labels[:, 0], iou_threshold=0.8
        )

    return pred_labels[idx_keep, :]


def decode_predicted_labels(predicted_labels):
    """Decode the S x S x 30 predicted tensor into a list of predicted objects.

    :param predicted_labels: The (N, S, S, 30) predictions tensor. The bounding
    box with the highest probability of contatining an object is kept.
    :type predicted_labels: torch.Tensor
    :return: The list of predicted objects
    :rtype: torch.Tensor

    """
    predicted_labels = predicted_labels.reshape(-1, 30)
    decoded_predicted_labels = torch.empty((predicted_labels.shape[0], 6))

    for cell, label in enumerate(predicted_labels):
        if label[0] > label[5]:
            decoded_predicted_labels[cell, :5] = label[:5]
        else:
            decoded_predicted_labels[cell, :5] = label[5:10]
        decoded_predicted_labels[cell, 5:] = torch.argmax(label[10:])

    return decoded_predicted_labels


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


def decode_labels(labels, S=7):
    """Decode the S x S x 5 target tensor to the target labels.

    :param labels: The encoded ground truth labels for the objects in the image
    :type labels: torch.Tensor
    :param S: Number of grid cells to split the image for in direction,
    defaults to 7
    :type S: int, optional
    :return: The ground truth labels for the objects in the image
    :rtype: torch.Tensor

    """
    # extract the labels from the grid cells containing an object
    # look for the width to be non-negative just in case that xc and yc have no
    # offsets relative to the grid cell
    grid_cells = torch.nonzero(labels[..., 2] > 0)
    labels = labels[grid_cells[:, 0], grid_cells[:, 1]]

    # convert centers of objects to be relative to the image rather than
    # being relative to the i,j grid cell
    labels[:, 0] = (labels[:, 0] + grid_cells[:, 0]) / S
    labels[:, 1] = (labels[:, 1] + grid_cells[:, 1]) / S

    return labels


def encode_labels(labels, S=7):
    """Encode the target labels into an S x S x 5 tensor.

    :param labels: The ground truth labels for the objects in the image
    :type labels: torch.Tensor
    :param S: Number of grid cells to split the image for in direction,
    defaults to 7
    :type S: int, optional
    :return: The encoded ground truth labels for the objects in the image
    :rtype: torch.Tensor

    """
    # divide the image into an S x S grid and assign an object to a grid cell
    # if the center of an object falls into that grid cell
    encoded_labels = torch.zeros((S, S, 5), dtype=labels.dtype)
    x, y = S * labels[:, 0], S * labels[:, 1]
    row = torch.floor(x).to(dtype=torch.long)
    col = torch.floor(y).to(dtype=torch.long)

    # the center coordinates of the object are relative to the grid cell and
    # the width and height are relative to the whole image
    # grid cells containing an object also contain the class of the object
    xc, yc = x - row, y - col
    labels[:, 0], labels[:, 1] = xc, yc
    encoded_labels[row, col, :] = labels

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


def voc_to_yolo_bbox(labels, im_shape):
    """Convert voc to yolo bounding box encoding.

    :param labels: The bouding box coordinates and class number for all of the
    objects in the image, [xmin, ymin, xmax, ymax, class_num]
    :type labels: numpy.ndarray
    :param im_shape: The height and width of the image
    :type im_shape: tuple of ints
    :return: The labels with the yolo bounding box encoding
    [xc, yc, w, h, class_num], where the coordinates are normalized to be
    relative to the entire image (between zero and one)
    :rtype: numpy.ndarray

    """
    labels[:, 2] -= labels[:, 0]        # width
    labels[:, 3] -= labels[:, 1]        # height
    labels[:, 0] += 0.5 * labels[:, 2]  # x_center
    labels[:, 1] += 0.5 * labels[:, 3]  # y_center
    labels[:, 0:-1:2] /= im_shape[0]
    labels[:, 1:-1:2] /= im_shape[1]

    return labels
