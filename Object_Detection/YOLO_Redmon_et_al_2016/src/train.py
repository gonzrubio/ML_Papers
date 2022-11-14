# -*- coding: utf-8 -*-
"""Train a YOLOv1 model on the custom VOC dataset.

Created on Sun Nov 13 14:18:59 2022

@author: gonzr
"""

import torch

from torch.utils.data import DataLoader

from dataset import VOCDetection
from evaluate import evaluate
from loss import YOLOv1Loss
from model import YOLO


def train(model, loss_fn, optim, epochs, train_dataloader, eval_dataloader):
    # TODO: save config, trained model and tensorboard plots
    # Tensorboard: loss(es), output on val set, validation metric(s),
    # activations at different layers and histogram weights
    #
    # save to directory:
    # YOLO_Redmond_et_al_2016/models/
    # ├─ run_name/
    # ├  ├── weights/
    # ├  ├── tensorboard/
    # ├  └── config.txt/json

    for epoch in range(epochs):

        loss_epoch = 0.
        for image, ground_truth in train_dataloader:
            prediction = model(image)
            loss = loss_fn(prediction, ground_truth)
            loss_epoch += loss.item()
            # compute gradient
            # step and update weights
        loss_epoch /= len(train_dataloader)

        if eval_dataloader:
            evaluate()

        # log in tensorboard


def main(config):

    model = YOLO()
    loss_fn = YOLOv1Loss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config['learning_rate'], momentum=0.9, weight_decay=0.0005
        )

    train_dataset = VOCDetection(split='train', train=True)
    train_dataloader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], collate_fn=train_dataset.collate_fn,
        pin_memory=config['pin_memory'], drop_last=config['drop_last'],
        prefetch_factor=config['prefetch_factor']
        )

    eval_dataloader = None
    if config['evaluate']:
        eval_dataset = VOCDetection(split='val', train=False)
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=config['batch_size'], shuffle=False,
            num_workers=config['num_workers'], pin_memory=config['pin_memory'],
            collate_fn=eval_dataset.collate_fn, drop_last=False,
            prefetch_factor=config['prefetch_factor']
            )

    train(
        model, loss_fn, optimizer, config['epochs'],
        train_dataloader, eval_dataloader
        )


if __name__ == "__main__":

    config = {
        'batch_size': 2,
        'shuffle': False,
        'num_workers': 1,
        'pin_memory': True,
        'drop_last': False,
        'prefetch_factor': 2,
        'evaluate': False,
        'optimizer': 'SGD',
        'learning_rate': 1e-4,
        'epochs': 100
        }

    main(*config)
