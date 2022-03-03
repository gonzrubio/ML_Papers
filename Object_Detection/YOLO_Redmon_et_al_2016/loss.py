"""You Only Look Once: Unified, Real-Time Object Detection.

Paper: https://arxiv.org/abs/1506.02640

Created on Tue Feb 8 21:38:56 2022

@author: gonzr
"""


import torch
import torch.nn as nn


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
        # Apply torch/Functional operations where possible

        # find the bounding box b responsible for detecting the object
        # by computing the confidence score of the detectors
        conf = torch.empty(size=(y_pred.shape[0], self.S, self.S, self.B))
        for b in range(self.B):
            conf[...,b:b+1] = IoU(y_pred[..., b*5+1:b*5+5], y_true[..., :-1])
            conf[...,b:b+1] *= y_pred[..., b*5:b*5+1]

        # confidence = ious * 
        detector = torch.max(conf, dim=-1)[1].unsqueeze(-1)

        # mask for grid cells that contain an object
        obj = (y_true[..., 0] > 0).int().unsqueeze(-1)

        # error for the bbox center
        # find way to construct the tensor with sampled detectors, maybe even retrace
        # to detector. would flat detector help index??
        loss = obj *  y_true[..., :2], y_pred[..., detector*5+1:detector*5+3]

        return obj


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    N = 1
    S = 7
    B = 2
    C = 20
    num_objects = 7

    loss = YOLOv1Loss(lambdas=[5, 0.5], S=S, B=B, C=C, reduction='sum')

    # dummy target batch, class is encoded as an integer in [1, num_classes]
    y_true = torch.zeros((N, S, S, 5))

    for batch in range(N):
        for obj in torch.randint(1, S**2, (num_objects,)):
            print(obj)
            x, y = obj // S, obj % S        
            y_true[batch, x, y, :-1] = torch.rand(4)
            y_true[batch, x, y, -1] = torch.randint(1, C, (1, ))

    
    print(loss(torch.rand((N, S, S, B * 5 + C)), y_true))
