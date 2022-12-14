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


def evaluate(model,
             dataloader,
             score_threshold,
             nms_threshold,
             iou_threshold,
             training=False):
    # """Evaluate the model on a dataset and compute the performance metrics.

    # :param model: The object detection model
    # :type model: torch.nn.Module
    # :param dataloader: The evaluation dataloader
    # :type dataloader: torch.utils.data.dataloader.DataLoader
    # :param training: If called from within the training loop, defaults to False
    # :type training: bool, optional
    # :return: The evaluation metrics
    # :rtype: tuple

    # """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    pred_all, gt_all = torch.tensor([]), torch.tensor([])

    model.to(device=device).eval()
    with torch.no_grad():
        for img_idx, sample in enumerate(tqdm(dataloader, desc='Evaluating')):
            image, gt, batch_idx = sample

            pred = model(image.to(device=device))
            pred = detect_objects(pred, score_threshold, nms_threshold)
            pred = torch.hstack(
                (torch.tensor([img_idx] * pred.shape[0]).unsqueeze(1), pred)
                )
            pred_all = torch.cat((pred, pred_all))

            gt = torch.hstack(
                (torch.tensor([img_idx] * gt.shape[0]).unsqueeze(1), gt)
                )
            gt_all = torch.cat((gt, gt_all))

    results = eval_metrics(pred_all, gt_all, iou_threshold=iou_threshold)

    if training:
        results['num_pred'] = pred_all.shape[0]
        results['num_gt'] = gt_all.shape[0]
    else:
        results['pred'] = pred_all.shape[0]
        results['gt'] = gt_all.shape[0]

    return results


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

    model = YOLO(fast=config['fast'])
    model.load_state_dict(checkpoint['model_state_dict'])

    results = evaluate(
        model, dataloader, config['score_threshold'], config['nms_threshold'],
        config['iou_threshold'], training=False
        )

    # plot precision-recall (implement in utils/plots, legend AP and mAP)
    # plot threshold-F1 (legend, F1 and mF1)
    # plot all predictions
    # save plots in figures/ (write directory structure in docstring)
    # color_palette = sns.color_palette()
    # fig = plt.figure(figsize=(10,7))
    # plt.plot(recall_curve, precision_curve,color=color_palette[2], lw=3)
    # plt.grid(True)
    # plt.xlabel("Recall")
    # plt.ylabel("Precision")
    # plt.title("Precision vs. Recall")
    # plt.xlim([.5,1.05])
    # plt.ylim([.5,1.05])
    # plt.locator_params(axis='x', nbins=11)
    # plt.locator_params(axis='y', nbins=11)

    #     plt.savefig(str(cfg["eval_directory"].joinpath("precision_recall_%d.pdf"%class_indx)))


if __name__ == "__main__":

    config = {
        'model': 'VOC_10',
        'fast': True,
        'dataset': 'VOC_10',
        'split': 'train',
        'score_threshold': 0.05,
        'nms_threshold': 0.9,
        'iou_threshold': 0.5,
        'num_workers': 0,
        'prefetch_factor': 2,
        }

    main(config)
