# -*- coding: utf-8 -*-
"""Evaluate a YOLOv1 model on a custom dataset.

Created on Sat Nov 26 13:20:35 2022

@author: gonzr
"""

import os
import torch

from torch.utils.data import DataLoader

from datasets import VOCDetection
from model import YOLO
from torchvision.ops import nms
from utils.bounding_boxes import decode_predicted_labels, yolo_to_voc_bbox


def evaluate(model, dataloader, device, training=False):
    # compute and return presicion, recall and F1
    # f1_score = 0
    for image, labels, batch_idx in dataloader:
        pred_labels = model(image.to(device=device))
        pred_labels = decode_predicted_labels(pred_labels)
        pred_labels = pred_labels[pred_labels[:, 0] > 0.4]
        idx_keep = nms(  # elements kept sorted by decreasing order of score
            boxes=yolo_to_voc_bbox(pred_labels[:, 1:5], (1, 1)),
            scores=pred_labels[:, 0], iou_threshold=0.5
            )
        pred_labels = pred_labels[idx_keep, :]

        # tp, fp, fn = tp_fp_fn(y, nms(pred_labels))
        # compute and return mAP
    return pred_labels


def main(config):
    dataset = VOCDetection(
        root=os.path.join('..', 'data', config['dataset']),
        split=config['split'], train=False, augment=False
        )

    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, pin_memory=True, drop_last=False,
        num_workers=config['num_workers'], collate_fn=dataset.collate_fn,
        prefetch_factor=config['prefetch_factor']
        )

    checkpoint = torch.load(
        os.path.join('..', 'models', config['model'], 'checkpoint.tar')
        )

    model = YOLO(fast=config['fast']).to(device=torch.device(config['device']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    results = evaluate(
        model, dataloader, torch.device(config['device']), training=False
        )

    # plot precision-recall
    # plot threshold-F1
    # plot predictions after nms (all or randomly selected)
    # save plots in figures/ (write directory structure in docstring)


if __name__ == "__main__":

    config = {
        'model': 'VOC_10',
        'dataset': 'VOC_10',
        'split': 'train',
        'fast': True,
        'num_workers': 0,
        'prefetch_factor': 2,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
        }

    main(config)
