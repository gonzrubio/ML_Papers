# -*- coding: utf-8 -*-
"""Train a YOLOv1 model on a custom VOC dataset.

Created on Sun Nov 13 14:18:59 2022

@author: gonzr
"""

import datetime
import json
import os
import torch

from torch.utils.data import DataLoader

from datasets import VOCDetection
from evaluate import evaluate
from loss import YOLOv1Loss
from model import YOLO


def train(
    model, loss_fn, optim, epochs, train_loader, eval_loader, device, save_dir
        ):

    torch.backends.cudnn.benchmark = True
    # scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):

        loss_epoch = 0.
        for batch_idx, (image, ground_truth) in enumerate(train_loader):

            # with torch.cuda.amp.autocast():
            #     prediction = model(image.to(device))
            #     loss = loss_fn(prediction, ground_truth.to(device))

            prediction = model(image.to(device))
            loss = loss_fn(prediction, ground_truth.to(device))

            loss_conf_obj, loss_coord, loss_class, loss_conf_noobj = loss
            loss = sum(loss)
            loss_epoch += loss.item()

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            # scaler.scale(loss).backward()
            # scaler.step(optim)
            # scaler.update()
            print(
                f"{epoch + 1}.{batch_idx + 1}",
                f"conf_obj: {loss_conf_obj.item():.4e}",
                f"coord: {loss_coord.item():.4e}",
                f"class: {loss_class.item():.4e}",
                f"conf_noobj: {loss_conf_noobj.item():.4e}",
                f"total: {loss.item():.4e}"
                )

        loss_epoch /= len(train_loader)

        if eval_loader:
            results = evaluate()
            num_gt, num_pred, tp, fp, fn, precision, recall, F1, mAP = results
            print(
                f"epoch: {epoch + 1} num_gt: {num_gt} num_pred: {num_pred}",
                f"recall: {recall:.4e} F1: {F1.item():.4e} ",
                f"mAP: {mAP.item():.4e}"
                )

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'loss': loss,
        'val': None,
        # 'val': mAP if eval_loader else None,
        }

    if eval_loader:
        print(checkpoint)  # append results
    torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.tar'))


def main(config):
    # saves to:
    # YOLO_Redmond_et_al_2016/
    # ├─ models/
    # ├  ├── run/
    # ├  ├   ├── checkpoint/
    # ├  ├   ├── tensorboard/
    # ├  ├   └── config.json

    run = '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())
    save_dir = os.path.join('..', 'models', run)
    os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'config.json'), 'w') as outfile:
        json.dump(config, outfile)

    model = YOLO(fast=config['fast']).to(device=torch.device(config['device']))
    loss_fn = YOLOv1Loss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config['learning_rate'], momentum=0.9, weight_decay=0.0005
        )

    train_dataset = VOCDetection(
        root=config['root'], split='train', train=True, augment=config['augment']
        )
    train_dataloader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], collate_fn=train_dataset.collate_fn,
        pin_memory=True, drop_last=config['drop_last'],
        prefetch_factor=config['prefetch_factor']
        )

    eval_dataloader = None
    if config['evaluate']:
        eval_dataset = VOCDetection(
            root=config['root'], split='val', train=False, augment=False
            )
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=config['batch_size'], shuffle=False,
            num_workers=config['num_workers'], pin_memory=True,
            collate_fn=eval_dataset.collate_fn, drop_last=False,
            prefetch_factor=config['prefetch_factor']
            )

    train(
        model, loss_fn, optimizer, config['epochs'], train_dataloader,
        eval_dataloader, torch.device(config['device']), save_dir
        )


if __name__ == "__main__":

    config = {
        'root': os.path.join('..', 'data', 'VOC_100'),
        'fast': True,
        'augment': False,
        'batch_size': 16,
        'shuffle': False,
        # 'num_workers': 2,
        'num_workers': 0,
        'drop_last': False,
        # 'prefetch_factor': 4,
        'prefetch_factor': 2,
        'evaluate': False,
        'optimizer': 'SGD',
        'learning_rate': 5e-4,
        'epochs': 4000,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
        }

    main(config)
