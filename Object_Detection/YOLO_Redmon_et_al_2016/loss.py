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

    def __init__(self, lambdas=[5., 0.5], reduction='sum'):
        """Construct the criterion.

        :param lambdas: DESCRIPTION, defaults to [5., 0.5]
        :type lambdas: TYPE, optional
        :param reduction: DESCRIPTION, defaults to 'sum'
        :type reduction: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE
        """
        super(YOLOv1Loss, self).__init__()

    def forward(self, x):
        """Apply the criterion to the predictions and ground truths.

        :param x: DESCRIPTION
        :type x: TYPE
        :return: DESCRIPTION
        :rtype: TYPE
        """
        # Apply the functional form
        pass


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
