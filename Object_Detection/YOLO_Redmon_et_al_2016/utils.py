"""common functionality."""


import torch


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
