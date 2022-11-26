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

from dataset import VOCDetection
# from evaluate import evaluate
from loss import YOLOv1Loss
from model import YOLO


def train(model, loss_fn, optim, epochs, train_loader, eval_loader, save_dir):

    device = next(model.parameters()).device
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

            loss_coord, loss_conf_obj, loss_conf_noobj, loss_class = loss
            loss = sum(loss)
            loss_epoch += loss.item()

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            # scaler.scale(loss).backward()
            # scaler.step(optim)
            # scaler.update()
            print(
                f"{epoch}.{batch_idx} ",
                f"{loss_coord.item():.4e}, {loss_conf_obj.item():.4e},",
                f"{loss_conf_noobj.item():.4e}, {loss_class.item():.4e}"
                )

        loss_epoch /= len(train_loader)

        # if eval_loader:
        #     mAP = evaluate()
        print(f"{epoch + 1} {loss_epoch:.4e}")
        # print(f"{epoch}.{batch_idx} {loss_epoch:.4e} {mAP:.4e}")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'loss': loss,
        'val': None,
        # 'val': mAP if eval_loader else None,
        }, os.path.join(save_dir, 'checkpoint'))


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
        pin_memory=config['pin_memory'], drop_last=config['drop_last'],
        prefetch_factor=config['prefetch_factor']
        )

    eval_dataloader = None
    if config['evaluate']:
        eval_dataset = VOCDetection(
            root=config['root'], split='val', train=False, augment=False
            )
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=config['batch_size'], shuffle=False,
            num_workers=config['num_workers'], pin_memory=config['pin_memory'],
            collate_fn=eval_dataset.collate_fn, drop_last=False,
            prefetch_factor=config['prefetch_factor']
            )

    train(
        model, loss_fn, optimizer, config['epochs'],
        train_dataloader, eval_dataloader, save_dir
        )


if __name__ == "__main__":

    # TODO lr schedule: 10x10-3, 73x10e-2, 26x10e-3, 26x10e-4
    config = {
        'root': '../data/VOC_10',
        'fast': True,
        'augment': False,
        # 'batch_size': 64,
        'batch_size': 16,
        # 'shuffle': True,
        'shuffle': False,
        # 'num_workers': 2,
        'num_workers': 0,
        'pin_memory': True,
        'drop_last': False,
        # 'prefetch_factor': 4,
        'prefetch_factor': 2,
        'evaluate': False,
        'optimizer': 'SGD',
        'learning_rate': 1e-2,  # TODO try 5e-3
        'epochs': 1000,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
        }

    main(config)
