# -*- coding: utf-8 -*-
"""Evaluation metrics for YOLO.

Created on Wed Nov 30 21:27:34 2022

@author: gonzr
"""

import torch

from utils.bounding_boxes import iou


def eval_metrics(
        predicted, ground_truth, img_idx_pred, img_idx_gt,
        iou_threshold=0.5, num_classes=20
        ):
    # pred_i = [prob, x_c, y_c, w, h, class_id]
    # gt_i = [x_c, y_c, w, h, class_id]
    # scores, precision_curve, recall_curve, F1_curve and threshold are
    # dictionaries with the different voc class lables as the keys and the
    # values are arrays with the sorted/cummulative values
    # the keys of precision_optimal, recall_optimal, F1_optimal and AP are
    # class labels and the values are the optimal value for that class, mF1
    # and mAP are the mean across the classes

    # TODO aladdinpersson mean_average_precision
    # TODO fizyr evaluate
    # TODO sgrvinod calculate_mAP

    results = {
        'scores': {}, 'precision_curve': {}, 'recall_curve': {}, 'F1_curve': {}, 'threshold': {},
        'precision': {}, 'recall': {}, 'F1': {}, 'AP': {}, 'mF1': None, 'mAP': None
        }

    scores_class, tp_class, fp_class, num_gt = match_pred_with_gt(
        predicted, ground_truth, img_idx_pred, img_idx_gt,
        iou_threshold=0.1, num_classes=num_classes
        )

    for class_idx in range(num_classes):
        if tp_class[class_idx].shape[0] == 0 or num_gt[class_idx] == 0:
            continue

        # sum all tp and fp
        tp_class[class_idx] = torch.cumsum(tp_class[class_idx])
        fp_class[class_idx] = torch.cumsum(fp_class[class_idx])

        # compute precision, recall and f1 score curves
        recall_curve = tp_class[class_idx] / num_gt[class_idx]
        precision_curve = fp_class[class_idx] / (tp_class[class_idx] + fp_class[class_idx] + 1e-9)
        F1_curve = 2 * precision_curve * recall_curve / (precision_curve + recall_curve + 1e-9)
        max_idx = torch.argmax(F1_curve)

        # compute average precision
        AP = None

        results['scores'][class_idx] = scores_class[class_idx]
        results['precision'][class_idx] = precision_curve
        results['recall'][class_idx] = recall_curve
        results['F1'][class_idx] = F1_curve
        results['threshold'][class_idx] = scores_class[class_idx][max_idx]
        results['precision'][class_idx] = precision_curve[max_idx]
        results['recall'][class_idx] = recall_curve[max_idx]
        results['F1'][class_idx] = F1_curve[max_idx]
        results['AP'][class_idx] = AP

    # compute mean for F1 and AP and make new results keys
    return results


def match_pred_with_gt(
        predicted, ground_truth, img_idx_pred, img_idx_gt,
        iou_threshold, num_classes
        ):

    predicted = torch.hstack((torch.tensor(img_idx_pred).unsqueeze(1), predicted))
    ground_truth = torch.hstack((torch.tensor(img_idx_gt).unsqueeze(1), ground_truth))
    scores_class = {}
    tp_class = {}
    fp_class = {}
    num_pred = {}
    num_gt = {}

    for class_idx in range(num_classes):

        # find predicted and ground truth detections for the class
        pred_class = predicted[predicted[:, -1] == class_idx]
        num_pred[class_idx] = pred_class.shape[0]

        gt_class = ground_truth[ground_truth[:, -1] == class_idx]
        # num_gt[class_idx] = gt_class.shape[0]
        num_gt[class_idx] = [0] * gt_class.shape[0]

        # sort the predictions in decreasing order of probability score
        sort_order = torch.argsort(-pred_class[:, 0])
        pred_class = pred_class[sort_order]

        # pre-allocate space for the class specific table
        scores_class[class_idx] = torch.zeros(num_pred[class_idx], 1)
        tp_class[class_idx] = torch.zeros(num_pred[class_idx], 1)
        fp_class[class_idx] = torch.zeros(num_pred[class_idx], 1)

        for ii, pred in enumerate(pred_class):
            scores_class[class_idx][ii] = pred[1]

            # if there are no gts then all detections are false positives
            if gt_class.shape[0] == 0:
                tp_class[class_idx][ii] = 0
                fp_class[class_idx][ii] = 1
                continue

            # otherwise try to assign a ground truth to the prediction (find
            # the gt with the highest iou from the same image)
            gt = gt_class[gt_class[:, 0] == pred[0]]
            overlaps = iou(pred[2:-1], gt[:, 1:-1])
            assigned_gt_bbox = torch.argmax(overlaps).item()
            max_iou = overlaps[assigned_gt_bbox]

            if max_iou >= iou_threshold:
                if True:  # fix here, only count once
                tp_class[class_idx][ii] = 1
                # detected_gt_bboxes.append(assigned_gt_bbox)
                else:
                    fp_class[class_idx][ii] = 1
            else:
                fp_class[class_idx][ii] = 1

    return scores_class, tp_class, fp_class, num_gt
