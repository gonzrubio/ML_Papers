"""You Only Look Once: Unified, Real-Time Object Detection.

Paper: https://arxiv.org/abs/1506.02640

Created on Tue Feb 8 21:38:56 2022

@author: gonzr
"""


import torch
import torch.nn as nn


class YOLOv1Loss(nn.Module):
    """Criterion that that frames object detection as a regression problem.

    This criterion is a multi-part loss function that measures the sum of
    squared errors between the predicted and target bounding boxes and their
    respective class label and conditional probability.
    """

    def __init__(self, lambdas, S=7, B=2, C=20, reduction='sum'):
        """Construct the criterion.

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
        # self.reduction = reduction
        self.mse = nn.MSELoss(reduction=reduction)

        # manual rescaling weights given to the sum of squared errors of
        # different components in the predicted and ground truth tensors
        self.lambda_obj_coord = lambdas[0]
        self.lambda_obj_p = 1
        self.lamba_noobj_coord = 0
        self.lambda_noobj_p = lambdas[1]
        self.lambda_c = 1

    def forward(self, x_pred, x_true):
        """Apply the criterion to the predictions and ground truths.

        :param x_pred: Predicted tensor of shape (N, S*S*(B*5+C)).
        :type x: torch.Tensor
        :param x_true: Ground truths of shape ????.
        :type x: torch.Tensor ?????
        :return: The computed loss between input and target. If `reduction` is
        `none`, shape (N) otherwise, scalar.
        :rtype: torch.Tensor
        """
        # Apply the functional form of operations where possible
        x_pred = x_pred.reshape(-1, self.S, self.S, self.B * 5 + self.C)
        # looks like doing mse of the output volume, look for efficient way
        # to compute the volume and check how the labels are stored sparse
        # format?

        # maybe the nms 
        # IoU for the B predicted and target bounding boxes
        return x_pred


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
