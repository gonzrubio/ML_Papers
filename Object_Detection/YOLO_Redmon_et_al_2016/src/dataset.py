"""Custom datasets for object detection.

Created on Fri Oct 28 13:33:06 2022

@author: gonzr
"""

import numpy as np
import os
import torch

from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.bounding_box import encode_labels

# from utils import plot_batch


class VOCDetection(Dataset):
    """Custom Pascal VOC dataset for object detection.

    The original 2007 and 2012 datasets are combined into a single dataset
    which is then randomly divided into train/val/test (70/20/10) splits.

    The combined Pascal VOC dataset has the following directory structure:

        VOC/
        ├─ Images/
        ├  ├── 1.jpg
        ├  ├── 2.jpg
        ├  └── n.jpg
        ├─ Annotations/
        ├  ├── 1.csv
        ├  ├── 2.csv
        ├  └── n.csv
        ├─ train.txt
        ├─ val.txt
        ├─ test.txt

    """

    def __init__(
            self,
            root='../data/VOC',
            split='train',
            train=True,
            transform=None
            ):
        """Parameters.

        :param root: Path to folder storing the dataset,
        defaults to '../data/VOC'
        :type root: str, optional
        :param split: Dataset split, one of 'train', 'val' or 'test',
        defaults to 'train'
        :type split: str, optional
        :param train: Training mode, returns the images and the encoded and
        augmented ground truth labels if True, otherwise it returns the raw
        ground truth labels, defaults to True
        :type train: bool, optional

        """
        self.images_dir = os.path.join(root, 'Images')
        self.annotations_dir = os.path.join(root, 'Annotations')
        with open(os.path.join(root, split + '.txt')) as f:
            self.data = f.read().splitlines()
        self.train = train
        # if self.train:
        #     self.transform = transforms.Compose(
        #         # random scaling, up to 20% translation, up to 1.5 times exposure
        #         # and saturation in hsv color space
        #         )

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get the (image, label) pair at the `idx` index."""
        filename = self.data[idx]
        image_path = os.path.join(self.images_dir, filename + '.jpg')
        label_path = os.path.join(self.annotations_dir, filename + '.csv')

        image = read_image(image_path)
        labels = torch.from_numpy(
            np.loadtxt(label_path, skiprows=1, delimiter=',')
            )

        if self.train:
            # TODO look into tranforms and how to use it
            image, labels = self.transform(image, labels)
            labels = encode_labels(labels)

        return image, labels


def voc_detection():

    # get batch in eval mode (img, labels)
    data = VOCDetection(split='train', train=False)
    dataloader = DataLoader(data, batch_size=64, shuffle=False)
    batch = next(iter(dataloader))
    # plot_batch(batch)  # color code class and write class name on bbox

    # get batch in train mode (img, encoded labels)
    # data = YOLO_VOC(split='train', train=True)
    # create data loader and get batch no shuffle so can compare same images
    # dataloader = DataLoader(data, batch_size=64, shuffle=False)
    # batch = next(iter(dataloader))
    # encoded labels to labels, decoding script in utils/bounding_box.py
    # plot_batch(batch)  # color code class and write class name on bbox


if __name__ == '__main__':

    voc_detection()
    # people_art()
