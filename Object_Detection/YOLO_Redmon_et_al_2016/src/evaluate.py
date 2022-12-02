# -*- coding: utf-8 -*-
"""Evaluate a YOLOv1 model on a custom dataset.

Created on Sat Nov 26 13:20:35 2022

@author: gonzr
"""

import os
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import VOCDetection
from model import YOLO
from utils.metrics import eval_metrics
from utils.bounding_boxes import detect_objects


def evaluate(model, dataloader, device, training=False):
    """Evaluate the model on a dataset and compute the performance metrics.

    :param model: The object detection model
    :type model: torch.nn.Module
    :param dataloader: The evaluation dataloader
    :type dataloader: torch.utils.data.dataloader.DataLoader
    :param device: Where to perform the computations
    :type device: torch.device
    :param training: If called from within the training loop, defaults to False
    :type training: bool, optional
    :return: The evaluation metrics
    :rtype: tuple

    """
    num_gt, num_pred = 0, 0
    pred_all, gt_all = torch.tensor([]), torch.tensor([])

    model.eval()
    with torch.no_grad():
        for ii, sample in enumerate(tqdm(dataloader, desc='Evaluating')):
            image, gt, batch_idx = sample
            pred = model(image.to(device=device))
            pred = detect_objects(pred, prob_threshold=0.15, iou_threshold=0.9)

            num_gt += len(gt)
            num_pred += len(pred)
            pred_all = torch.cat((pred, pred_all))
            gt_all = torch.cat((gt, gt_all))

    results = eval_metrics(pred_all, gt_all)

    if training:
        return num_gt, num_pred, results

    return pred_all, gt_all, results


def main(config):
    """Evaluate the model and save the results."""
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
    # pretty print AP and mAP
    # plot all predictions
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
