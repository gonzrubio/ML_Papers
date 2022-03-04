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
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    box1_x1 = bbox_pred[..., 0:1] - bbox_pred[..., 2:3] / 2
    box1_y1 = bbox_pred[..., 1:2] - bbox_pred[..., 3:4] / 2
    box1_x2 = bbox_pred[..., 0:1] + bbox_pred[..., 2:3] / 2
    box1_y2 = bbox_pred[..., 1:2] + bbox_pred[..., 3:4] / 2

    box2_x1 = bbox_pred[..., 0:1] - bbox_pred[..., 2:3] / 2
    box2_y1 = bbox_pred[..., 1:2] - bbox_pred[..., 3:4] / 2
    box2_x2 = bbox_pred[..., 0:1] + bbox_pred[..., 2:3] / 2
    box2_y2 = bbox_pred[..., 1:2] + bbox_pred[..., 3:4] / 2

    x1, y1 = torch.max(box1_x1, box2_x1), torch.max(box1_y1, box2_y1)
    y2, x2 = torch.min(box1_x2, box2_x2), torch.min(box1_y2, box2_y2)

    # set to zero if the bounding boxes do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


class YOLOv1Loss(nn.Module):
    """Criterion that that frames object detection as a regression problem.

    This criterion is a multi-part loss function that measures the squared
    errors between the predicted and target bounding boxes, their respective
    confidence scores and the conditional class probability.
    """

    def __init__(self, lambdas=[5, 0.5], S=7, B=2, C=20, reduction='sum'):
        """Construct the criterion.

        # manual rescaling weights given to the sum of squared errors of
        # different components in the predicted and ground truth tensors
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

        :param x_pred: Predicted tensor of shape (N, S*S*(B*5+C)).
        :type x: torch.Tensor
        :param x_true: Ground truths of shape ????.
        :type x: torch.Tensor ?????
        :return: The computed loss between input and target. If `reduction` is
        `none`, shape (N) otherwise, scalar.
        :rtype: torch.Tensor
        """
        #  mask_obj = y_true > 0   # maintains the same shape as y_true
        # produces shape [num_objects, 5 | S*S*(B*5+C)]
        mask_obj = (y_true > 0)[...,0]
        mask_noobj = torch.logical_not(mask_obj)

        # y_true_obj = y_true[mask_obj].reshape(-1, 5)   # [num_objects, 5]
        y_true_obj = y_true[mask_obj]       # [num_objects, 5]
        # y_true_noobj = y_true[mask_noobj]   # [S^2 - num_objects, 5]

        # y_pred_obj = y_pred[mask_obj[..., 0]]
        y_pred_obj = y_pred[mask_obj]       # [num_objects, S*S*(B*5+C)]
        y_pred_noobj = y_pred[mask_noobj]   # [S^2 - num_objects, 5]

        # now need to find max IoU between all B bbouning boxes and the targets
        # returns shape [num_objects, len([conf_score, cx, cy, w, h]) + C]    
        y_pred_obj = self._maxIoU(y_true_obj, y_pred_obj)


        # loss coord
        y_true_obj[:, 2:4] = torch.sqrt(y_true_obj[:, 2:4])
        y_pred_obj[:, 3:5] = torch.sqrt(y_pred_obj[:, 3:5] + 1e-6)

        lambda_coord = 5
        loss_coord = lambda_coord * F.mse_loss(
            y_pred_obj[:, 1:5], y_true_obj[:, :4], reduction='sum'
            )


        # loss confidence scores
        loss_conf_obj = F.mse_loss(
            torch.ones_like(y_pred_obj[:, 0]), y_pred_obj[:, 0], reduction='sum'
            )

        lambda_noobj = 0.5
        y_pred_noobj = y_pred_noobj[:, :-C].reshape(-1, 5)[:, 0]  # [-1, 5]

        loss_conf_noobj = lambda_noobj * F.mse_loss(
            torch.zeros_like(y_pred_noobj), y_pred_noobj, reduction='sum'
            )


        # one hot encoding loss
        loss_class = F.mse_loss(
            F.one_hot(y_true_obj[:, -1].long(), num_classes=20),
            y_pred_obj[..., -C:]
            )


        # total loss
        loss = loss_coord + loss_conf_obj + loss_conf_noobj + loss_class

        # # Apply torch/Functional operations where possible

        # # find the bounding box b responsible for detecting the object
        # # by computing the confidence score of the detectors
        # conf = torch.empty(size=(y_pred.shape[0], self.S, self.S, self.B))
        # for b in range(self.B):
        #     conf[...,b:b+1] = IoU(y_pred[..., b*5+1:b*5+5], y_true[..., :-1])
        #     conf[...,b:b+1] *= y_pred[..., b*5:b*5+1]

        # # confidence = ious * 
        # detector = torch.max(conf, dim=-1)[1].unsqueeze(-1)

        # # mask for grid cells that contain an object
        # obj = (y_true[..., 0] > 0).int().unsqueeze(-1)

        # # error for the bbox center
        # # find way to construct the tensor with sampled detectors, maybe even retrace
        # # to detector. would flat detector help index??
        # loss = obj *  y_true[..., :2], y_pred[..., detector*5+1:detector*5+3]

        return loss

    def _maxIoU(y_true_obj, y_pred_obj):
        return y_pred_obj[..., 0: 5+ 20]


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    N = 1
    S = 7
    B = 2
    C = 20
    num_objects = 7

    # dummy target batch [N, S, S, len([center_x, center_y, w, h, class_num])]
    # and dummy predicted volume
    y_true = torch.zeros((N, S, S, 5))
    y_pred = torch.rand((N, S, S, B * 5 + C))

    for b in range(N):
        for obj in torch.randint(1, S**2, (num_objects,)):
            cell = random.randint(0, S**2 - 1)
            row, col = cell // 7, cell % 7
            y_true[b, row, col, :-1] = torch.rand(size=(4, ))
            y_true[b, row, col, -1] = torch.randint(low=1, high=C, size=(1, ))

    
    loss = YOLOv1Loss(lambdas=[5, 0.5], S=S, B=B, C=C, reduction='sum')

    print(loss(y_pred, y_true))
