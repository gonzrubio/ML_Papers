# -*- coding: utf-8 -*-
"""Evaluate a YOLOv1 model on a custom dataset.

Created on Sat Nov 26 13:20:35 2022

@author: gonzr
"""

import os
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from data.make_voc_dataset import ID_CLASS_MAP, ID_COLOR_MAP
from datasets import VOCDetection
from model import YOLO
from utils.bounding_boxes import detect_objects
from utils.metrics import eval_metrics
from utils.plots import plot_gt_vs_pred, plot_AP_F1


def evaluate(model,
             dataloader,
             score_threshold,
             nms_threshold,
             iou_threshold,
             training=False):
<<<<<<< HEAD
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

=======
    """Evaluate the model on a dataset and compute the performance metrics.

    Parameters
    ----------
    model : torch.nn.Module
        The object detection model.
    dataloader : torch.utils.data.dataloader.DataLoader
        The evaluation dataloader.
    score_threshold : float
        The probability threshold for detections to keep.
    nms_threshold : float
        Threshold for nms to remove detections above that threshold.
    iou_threshold : float
        The iou threshold to count detection as TP.
    training : bool, optional
        If called from within the training loop, defaults to False. The default
        is False.

    Returns
    -------
    results : tuple
        The evaluation metrics.

    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device=device).eval()

    pred_all, gt_all = torch.tensor([]), torch.tensor([])
    with torch.no_grad():
        for img_idx, sample in enumerate(tqdm(dataloader, desc='Evaluating')):
            image, gt = sample[:2]

            pred = model(image.to(device=device))
            pred = detect_objects(pred, score_threshold, nms_threshold)
            if pred.shape[0] > 0:
                pred = torch.hstack(
                    (torch.tensor([img_idx] * pred.shape[0]).unsqueeze(1), pred)
                    )
                pred_all = torch.cat((pred_all, pred))

                gt = torch.hstack(
                    (torch.tensor([img_idx] * gt.shape[0]).unsqueeze(1), gt)
                    )
                gt_all = torch.cat((gt_all, gt))

>>>>>>> yolov1
    results = eval_metrics(pred_all, gt_all, iou_threshold=iou_threshold)

    if training:
        results['num_pred'] = pred_all.shape[0]
        results['num_gt'] = gt_all.shape[0]
    else:
<<<<<<< HEAD
        results['pred'] = pred_all.shape[0]
        results['gt'] = gt_all.shape[0]
=======
        results['pred'] = pred_all
        results['gt'] = gt_all
>>>>>>> yolov1

    return results


def main(config):
    """Evaluate a model and save the results.


    The generated plots are saved to:

    YOLO_Redmond_et_al_2016/
    ├─ plots/
    ├  ├── model/
    ├  ├   ├── pred_vs_gt/
    ├  ├   ├── F1_confidence.png
    ├  ├   └── precision_recall_AP.png

    """
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

    save_dir = os.path.join('..', 'plots', config['model'])
    save_dir_pred_gt = os.path.join(save_dir, 'pred_vs_gt')
    os.makedirs(save_dir_pred_gt)
    plot_gt_vs_pred(dataloader,
                    results['pred'],
                    ID_CLASS_MAP,
                    ID_COLOR_MAP,
                    size=(896, 896),
                    fill=True,
                    save_dir=save_dir_pred_gt)
    plot_AP_F1(results, ID_CLASS_MAP, ID_COLOR_MAP, save_dir=save_dir)



if __name__ == "__main__":

    config = {
<<<<<<< HEAD
        'model': 'VOC_10',
        'fast': True,
        'dataset': 'VOC_10',
        'split': 'train',
        'score_threshold': 0.05,
        'nms_threshold': 0.9,
        'iou_threshold': 0.5,
        'num_workers': 0,
        'prefetch_factor': 2,
=======
        'model': 'YOLO_fast_100_samples',
        'fast': True,
        'dataset': 'VOC_100',
        'split': 'train',
        'num_workers': 0,
        'prefetch_factor': 2,
        'score_threshold': 0.4,
        'nms_threshold': 0.7,
        'iou_threshold': 0.5,
>>>>>>> yolov1
        }

    main(config)
