"""You Only Look Once: Unified, Real-Time Object Detection.

Paper: https://arxiv.org/abs/1506.02640

Created on Tue Feb 8 21:38:56 2022

@author: gonzr
"""


import random
import torch
import torch.nn as nn
import torch.nn.functional as F


def IoU(bbox_pred, bbox_true):
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


class YOLOv1Loss(nn.Module):
    """Criterion that that frames object detection as a regression problem.

    This criterion is a multi-part loss function that measures the squared
    errors between the predicted and target bounding boxes, their respective
    confidence scores and the conditional class probability.
    """

    def __init__(self, lambdas=[5, 0.5], S=7, B=2, C=20, reduction='sum'):
        """Construct the criterion.

        :param lambdas: Manual rescaling weights given to the loss from
        bounding box coordinate predictions and for boxes that don't contain
        objects, defaults to [5, 0.5]
        :type lambdas: list, optional
        :param S: Number of grid cells to split the image for each direction,
        defaults to 7
        :type S: int, optional
        :param B: Number of predicted bounding boxes per gird cell,
        defaults to 2
        :type B: int, optional
        :param C: Number of class labels, defaults to 20
        :type C: int, optional
        :param reduction: pecifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed.
            Defaults to 'sum'.
        :type reduction: str, optional
        """
        super(YOLOv1Loss, self).__init__()        
        self.S = S
        self.B = B
        self.C = C
        self.reduction = reduction
        self.lambda_coord, self.lamba_noobj = lambdas

    def forward(self, y_pred, y_true):
        """Apply the criterion to the predictions and ground truths.

        :param y_pred: Predicted tensor of shape (N, S, S, B * 5 + C).
        :type y_pred: torch.Tensor
        :param y_true: Ground truths of shape (N, S, S, 5 + C), the normalized
        4-dimensional bounding box center, width and height, and the class id.
        :type y_true: torch.Tensor
        :return: The computed loss between input and target. If `reduction` is
        `none`, shape (N) otherwise, scalar.
        :rtype: torch.Tensor
        """
        # produces shape [num_objects, 5 | S*S*(B*5+C)]
        mask_obj = (y_true > 0)[..., -1]
        mask_noobj = torch.logical_not(mask_obj)

        y_true_obj = y_true[mask_obj]       # [num_objects, 5]
        y_pred_obj = y_pred[mask_obj]       # [num_objects, S*S*(B*5+C)]
        y_pred_noobj = y_pred[mask_noobj]   # [S^2 - num_objects, 5]

        # finds detector with the highest confidence score between all 
        # bouning boxes and targets.
        y_pred_obj = self._max_confidence_score(y_true_obj, y_pred_obj)

        # loss coord
        y_true_obj[:, 2:4] = torch.sqrt(y_true_obj[:, 2:4])
        y_pred_obj[:, 3:5] = torch.sqrt(y_pred_obj[:, 3:5])

        lambda_coord = 5
        loss_coord = lambda_coord * F.mse_loss(
            y_pred_obj[:, 1:5], y_true_obj[:, :4], reduction=self.reduction
            )

        # loss confidence scores
        loss_conf_obj = F.mse_loss(
            torch.ones_like(y_pred_obj[:, 0]), y_pred_obj[:, 0], reduction=self.reduction
            )

        lambda_noobj = 0.5
        y_pred_noobj = y_pred_noobj[:, :-C].reshape(-1, 5)[:, 0]  # [empty cells * B]

        loss_conf_noobj = lambda_noobj * F.mse_loss(
            torch.zeros_like(y_pred_noobj), y_pred_noobj, reduction=self.reduction
            )

        # one hot encoding loss
        loss_class = F.mse_loss(
            F.one_hot(y_true_obj[:, -1].long() - 1, num_classes=self.C),
            y_pred_obj[..., -C:],
            reduction=self.reduction
            )

        loss = loss_coord + loss_conf_obj + loss_conf_noobj + loss_class
        return loss / float(y_true.shape[0])

    def _max_confidence_score(self, y_true_obj, y_pred_obj):
        """Find the bounding box b responsible for detecting the object.

        :param y_true_obj: Grid cells containing an object [num_objects, 5]
        :type y_true_obj: torch.Tensor
        :param y_pred_obj: Predictions for grid cells containing an object [num_objects, B*5+C]
        :type y_pred_obj: torch.Tensor
        :return: y_pred_obj of shape [num_objects, len([conf_score, cx, cy, w, h]) + C]    
        :rtype: torch.Tensor
        """
        conf = torch.empty((y_pred_obj.shape[0], self.B), device=y_pred_obj.device)
        for b in range(self.B):
            conf[..., b:b+1] = IoU(y_pred_obj[..., b*5+1:b*5+5], y_true_obj[..., :-1])
            conf[..., b:b+1] *= y_pred_obj[..., b*5:b*5+1]

        # mask for bounding box with highest confidence score 
        mask_predictor = torch.zeros_like(y_pred_obj, dtype=torch.bool)
        for row, bbox in enumerate(torch.max(conf, dim=-1)[1]):
            mask_predictor[row, bbox:bbox+5] = True
            mask_predictor[row, -self.C:] = True

        return y_pred_obj[mask_predictor].reshape(-1, 5 + self.C)


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    N = 2**3
    S = 7
    B = 2
    C = 20
    num_objects = 13

    # dummy target batch [N, S, S, len([center_x, center_y, w, h, class_num])]
    # and dummy predicted volume [N, S, S, S*S*(B*5+C)]
    y_true = torch.zeros((N, S, S, 5), device=device)
    # y_pred = torch.rand((N, S, S, B * 5 + C), device=device)
    y_pred = torch.zeros((N, S, S, B * 5 + C), device=device) # perfect prediction

    for b in range(N):
        for obj in torch.randint(1, S**2, (num_objects,)):
            cell = random.randint(0, S**2 - 1)
            row, col = cell // S, cell % S
            y_true[b, row, col, :-1] = torch.rand(size=(4, ))
            y_true[b, row, col, -1] = torch.randint(low=1, high=C, size=(1, ))

            # perfect prediction
            bbox = torch.randint(low=0, high=B-1, size=(1, ))
            y_pred[b, row, col, bbox * 5] = 1.
            y_pred[b, row, col, bbox * 5 + 1:bbox * 5 + 5] = y_true[b, row, col, :-1]
            y_pred[b, row, col, -C:] = F.one_hot(y_true[b, row, col, -1].long()-1, num_classes=C)

    
    loss = YOLOv1Loss(lambdas=[5, 0.5], S=S, B=B, C=C, reduction='sum')

    print(loss(y_pred, y_true))
