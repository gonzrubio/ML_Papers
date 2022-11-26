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


def evaluate(model, dataloader, device, training=False):
    # use batches with mode sum reduce and divide by num samples
    # compute and return mAP
    # compute and return presicion, recall and F1
    pass


def main(config):
    dataset = VOCDetection(
        root=os.path.join('..', 'data', config['dataset']),
        split=config['split'], train=False, augment=False
        )

    dataloader = DataLoader(
        dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=config['pin_memory'],
        collate_fn=dataset.collate_fn, drop_last=False,
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
        'batch_size': 16,
        'num_workers': 0,
        'pin_memory': True,
        'drop_last': False,
        'prefetch_factor': 2,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
        }

    main(config)
