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

from utils.bounding_boxes import detect_objects


def evaluate(model, dataloader, device, training=False):
    # compute num_gt, num_pred, tp, fp, fn, precision, recall, F1, mAP

    num_gt = 0
    num_pred = 0

    model.eval()
    with torch.no_grad():
        for image, labels, batch_idx in dataloader:
            pred_labels = model(image.to(device=device))
            pred_labels = detect_objects(pred_labels)
            num_gt += len(labels)
            num_pred += len(pred_labels)
            

    model.train()
    return pred_labels
    # return num_gt, num_pred, tp, fp, fn, precision, recall, F1, mAP


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

    results = evaluate(
        model, dataloader, torch.device(config['device']), training=False
        )

    # plot precision-recall
    # plot threshold-F1
    # plot predictions (all or randomly selected)
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
