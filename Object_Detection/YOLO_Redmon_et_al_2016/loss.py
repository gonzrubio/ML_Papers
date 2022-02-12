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

    def __init__(self, lambdas, reduction='sum'):
        """Construct the criterion.

        :param lambdas: A manual rescaling weight given to the loss for
        bounding box coordinate predictions and to the loss for boxes that
        don't contain objects, i.e. torch.Tensor([5., .5], device=DEVICE).
        :type lambdas: list
        :param reduction: pecifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed.
            Defaults to 'sum'.
        :type reduction: str, optional
        """
        super(YOLOv1Loss, self).__init__()
        self.lambdas = lambdas
        self.reduction = reduction

    def forward(self, x):
        """Apply the criterion to the predictions and ground truths.

        :param x: Predicted probabilities of shape (N, 1470).
        :type x: torch.Tensor
        :return: The computed loss between input and target. If `reduction` is
        `none`, shape (N) otherwise, scalar.
        :rtype: torch.Tensor
        """
        # Apply the functional form
        pass


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
