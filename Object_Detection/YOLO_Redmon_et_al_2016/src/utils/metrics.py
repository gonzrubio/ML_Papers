# -*- coding: utf-8 -*-
"""Evaluation metrics for YOLO.

Created on Wed Nov 30 21:27:34 2022

@author: gonzr
"""

import torch

from utils.bounding_boxes import iou


def eval_metrics(pred, gt, iou_threshold=0.5, num_classes=20):
    """Compute object detection metrics for a set of predictions and gt.

    :param pred: A n x 7 tensor [image number, score, cx, cy, w, h, class]
    :type pred: torch.Tensor
    :param gt: A m x 6 tensor [image number, cx, cy, w, h, class]
    :type gt: torch.Tensor
    :param iou_threshold: Minimum iou to associate a prediction with a gt,
    defaults to 0.5
    :type iou_threshold: float, optional
    :param num_classes: Number of object classes in the dataset, defaults to 20
    :type num_classes: int, optional
    :return: Object detection metrics per class for a set of predictions and gt
    :rtype: dict

    """
    results = {
        'scores': {}, 'precision_curve': {}, 'recall_curve': {},
        'F1_curve': {}, 'threshold': {}, 'precision': {}, 'recall': {},
        'F1': {}, 'AP': {}, 'mF1': None, 'mAP': None
        }

    scores_class, tp_class, fp_class, num_gt_class = match_pred_with_gt(
        pred, gt, iou_threshold=iou_threshold, num_classes=num_classes
        )

    for class_idx in range(num_classes):
        if tp_class[class_idx].shape[0] == 0 or num_gt_class[class_idx] == 0:
            continue

        # sum all tp and fp
        tp_class[class_idx] = torch.cumsum(tp_class[class_idx], dim=0)
        fp_class[class_idx] = torch.cumsum(fp_class[class_idx], dim=0)

        # compute precision, recall and f1 score curves
        recall_curve = tp_class[class_idx] / num_gt_class[class_idx]
        precision_curve = fp_class[class_idx] / (
            tp_class[class_idx] + fp_class[class_idx] + 1e-9
            )
        F1_curve = 2 * precision_curve * recall_curve / (
            precision_curve + recall_curve + 1e-9
            )
        max_idx = torch.argmax(F1_curve)

        # compute average precision
        precision_curve = torch.vstack((torch.tensor([1]), precision_curve))
        recall_curve = torch.vstack((torch.tensor([0]), recall_curve))
        AP = torch.trapz(precision_curve, recall_curve, dim=0).item()

        results['scores'][class_idx] = scores_class[class_idx]
        results['precision_curve'][class_idx] = precision_curve
        results['recall_curve'][class_idx] = recall_curve
        results['F1_curve'][class_idx] = F1_curve
        results['threshold'][class_idx] = scores_class[class_idx][max_idx].item()
        results['precision'][class_idx] = precision_curve[max_idx].item()
        results['recall'][class_idx] = recall_curve[max_idx].item()
        results['F1'][class_idx] = F1_curve[max_idx].item()
        results['AP'][class_idx] = AP

    # compute mean for F1 and AP
    n_seen_classes = sum([1 if i[1] > 0 else 0 for i in num_gt_class.items()])
    results['mAP'] = sum(results['AP'].values()) / n_seen_classes
    results['mF1'] = sum(results['F1'].values()) / n_seen_classes

    return results


def match_pred_with_gt(predicted, ground_truth, iou_threshold, num_classes):
    """Associate predicted objects with the ground truths.

    :param predicted: A n x 7 tensor [image number, score, cx, cy, w, h, class]
    :type predicted: torch.Tensor
    :param ground_truth: A m x 6 tensor [image number, cx, cy, w, h, class]
    :type ground_truth: torch.Tensor
    :param iou_threshold: Minimum iou to associate a prediction with a gt
    :type iou_threshold: float
    :param num_classes: Number of object classes in the dataset
    :type num_classes: int
    :return: class scores_class, tp, fp, and number of ground truths
    :rtype: tuple

    """
    scores_class = {}
    tp_class = {}
    fp_class = {}
    num_gt_class = {}

    for class_idx in range(num_classes):

        # find predicted and ground truth detections for the class
        pred_class = predicted[predicted[:, -1] == class_idx]
        gt_class = ground_truth[ground_truth[:, -1] == class_idx]
        num_gt_class[class_idx] = gt_class.shape[0]

        # sort the predictions in decreasing order of probability score
        sort_idx = torch.argsort(-pred_class[:, 1])
        pred_class = pred_class[sort_idx, :]

        # pre-allocate space for the class specific table
        scores_class[class_idx] = pred_class[:, 1]
        tp_class[class_idx] = torch.zeros(pred_class.shape[0])
        fp_class[class_idx] = torch.zeros(pred_class.shape[0])

        # if there are no gts then all detections are false positives
        if num_gt_class[class_idx] == 0:
            fp_class[class_idx] += 1
            continue

        # keep track of assigned gt per class and image number
        seen = {}
        for image_number in torch.unique(gt_class[:, 0]):
            num_objects = gt_class[gt_class[:, 0] == image_number].shape[0]
            seen[image_number.item()] = [False for i in range(num_objects)]

        for pred_idx, pred in enumerate(pred_class):
            img_number = pred[0].item()
            gt = gt_class[gt_class[:, 0] == img_number]

            if gt.shape[0] > 0:
                overlaps = iou(pred[2:-1], gt[:, 1:-1])
                assigned_gt = torch.argmax(overlaps).item()
                max_iou = overlaps[assigned_gt]

                if max_iou >= iou_threshold and not seen[img_number][assigned_gt]:
                    tp_class[class_idx][pred_idx] = 1
                    seen[img_number][assigned_gt] = True
                else:
                    fp_class[class_idx][pred_idx] = 1
            else:
                fp_class[class_idx][pred_idx] = 1

    return scores_class, tp_class, fp_class, num_gt_class
