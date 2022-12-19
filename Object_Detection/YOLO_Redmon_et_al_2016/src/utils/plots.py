# -*- coding: utf-8 -*-
"""Functionality to visualize the data.

Created on Mon Oct 31 16:03:13 2022

@author: gonzr
"""

import matplotlib.pyplot as plt
import os
import torch

from torchvision.utils import draw_bounding_boxes
from utils.bounding_boxes import yolo_to_voc_bbox, decode_labels
from utils.transforms import InvToTensorNormalize


def plot_AP_F1(results, id_class_map, id_color_map, save_dir):
    """Generate precision-recall and F1-condicence plots.

    Parameters
    ----------
    results : dict
        The model evaliuation results.
    id_class_map : dict
        A map from class number to text ie. 0 -> 'aeroplane'.
    id_color_map : dict
        Amap from class number to color for the bounding boxes.
    save_dir : str, optional
        If not None, the directory where the generated plots are saved to.

    Returns
    -------
    None.

    """
    legend = []
    for class_number in range(20):
        try:
            x = results['recall_curve'][class_number]
            y = results['precision_curve'][class_number]
            plt.plot(x, y)
            legend.append(f'{id_class_map[class_number]}: ' \
                          f'{results["AP"][class_number]: .4f} AP')
        except:
            continue
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_prop_cycle(color=id_color_map.values())
    plt.legend(legend, loc='upper right', fontsize=6.5)
    plt.savefig(os.path.join(save_dir, 'precision_recall_AP.png'),
                dpi=1200,
                bbox_inches='tight',
                pad_inches=0)
    plt.show()

    legend = []
    for class_number in range(20):
        try:
            x = results['scores'][class_number]
            y = results['F1_curve'][class_number]
            plt.plot(x, y)
            legend.append(f'{id_class_map[class_number]}: ' \
                          f'{results["F1"][class_number]: .4f} F1 score')
        except:
            continue
    plt.xlabel('confidence')
    plt.ylabel('F1')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_prop_cycle(color=id_color_map.values())
    plt.legend(legend, loc='upper right', fontsize=6.5)
    plt.savefig(os.path.join(save_dir, 'F1_confidence.png'),
                dpi=1200,
                bbox_inches='tight',
                pad_inches=0)
    plt.show()


def plot_gt_vs_pred(dataloader,
                    preds,
                    id_class_map,
                    id_color_map,
                    size=(448, 448),
                    fill=True,
                    save_dir=None):
    """Generate side-by-side comparisson of ground truth vs predictions.

    Parameters
    ----------
    dataloader : torch.utils.data.dataloader.DataLoader
        The dataloader containing the dataset images.
    preds : torch.Tensor
        A [num_preds, 7] tensor with the predicted detections.
    id_class_map : dict
        A map from class number to text ie. 0 -> 'aeroplane'.
    id_color_map : dict
        Amap from class number to color for the bounding boxes.
    size : tuple, optional
        The plot size. The default is (448, 448).
    fill : bool, optional
        Shade in the bounding boxes. The default is True.
    save_dir : str, optional
        If not None, the directory where the generated plots are saved to.

    Returns
    -------
    None.

    """
    for img_idx, (image, gt, batch_idx) in enumerate(dataloader):
        plot_batch(
                (image, gt, batch_idx),
                id_class_map,
                id_color_map,
                size=size,
                fill=fill,
                save_dir=os.path.join(save_dir, f'{img_idx}_gt.png')
                )
        pred = preds[preds[:, 0] == img_idx][:, 2:]
        plot_batch(
                (image, pred, torch.zeros(pred.shape[0])),
                id_class_map,
                id_color_map,
                size=size,
                fill=fill,
                save_dir=os.path.join(save_dir, f'{img_idx}_pred.png')
                )

def plot_batch(
        batch,
        id_class_map,
        id_color_map,
        size=(448, 448),
        fill=True,
        save_dir=None
        ):
    """Plot a collated batch of (image, labels) pairs.

    :param batch: A collated batch of image-label pairs. One of:
        - A batched image tensor and a batched labels tensor.
        - A batched image tensor, a stacked labels tensor and a tensor of batch
        indices mapping each bounding box to its respective image in the batch.
    :type batch: tuple
    :param size: The size to resize the images to, defaults to (448, 448)
    :type size: TYPE, optional
    :param fill: Shade in the bounding boxes, defaults to 'true'
    :type fill: str, optional
    :param save_dir: If specified, the directory to save the plotted images to,
    it can be a directory or in the case of a single image the full file name.
    defaults to None
    :type save_dir: str, optional

    """
    if batch[-1].dim() == 4:
        mode = 'train'
        (images, labels) = batch
    elif batch[-1].dim() == 1:
        mode = 'eval'
        (images, labels, batch_idx) = batch

    inverse_transform = InvToTensorNormalize(size)

    for idx, image in enumerate(images):

        image = inverse_transform(image).to(dtype=torch.uint8)

        if mode == 'train':
            labels_image = decode_labels(labels[idx])
        elif mode == 'eval':
            labels_image = labels[batch_idx == idx]

        boxes = labels_image[:, :-1]
        boxes = yolo_to_voc_bbox(boxes, (image.shape[-2], image.shape[-1]))

        classes = [None] * len(labels_image)
        colors = [None] * len(labels_image)
        for box, class_id in enumerate(labels_image[:, -1]):
            classes[box] = id_class_map[int(class_id.item())]
            colors[box] = id_color_map[int(class_id.item())]

        kwargs = {
            'labels': classes, 'colors': colors, 'fill': fill, 'width': 4,
            }
        drawn_boxes = draw_bounding_boxes(image, boxes, **kwargs)
        drawn_boxes = drawn_boxes.permute(1, 2, 0).numpy()

        plt.imshow(drawn_boxes)
        plt.axis("off")
        plt.show()

        if save_dir.endswith('.png'):
            plt.imsave(save_dir, drawn_boxes, dpi=1200)
        else:
            plt.imsave(os.path.join(save_dir, f'{idx}.png'), drawn_boxes, dpi=1200)
