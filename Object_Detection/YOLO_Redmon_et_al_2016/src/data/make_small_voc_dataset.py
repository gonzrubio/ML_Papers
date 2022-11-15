# -*- coding: utf-8 -*-
"""Create a small training dataset from the custom VOC dataset.

Created on Mon Nov 14 19:12:46 2022

@author: gonzr


This script creates a small training dataset for prototyping and debugging. The
directory of the small dataset has the following structure:
    data/VOC_${num_samples}/
    ├─ Images/
    ├  ├── 1.jpg
    ├  ├── 2.jpg
    ├  └── m.jpg
    ├─ Annotations/
    ├  ├── 1.csv
    ├  ├── 2.csv
    ├  └── m.csv
    ├─ train.txt

The samples are randomly copied from the custom VOC dataset.

"""

import glob
import os
import random
import shutil


def main(dataset_path, num_samples, new_path):
    """Create a smaller dataset from the custom VOC dataset.

    :param dataset_path: The path to the full custom VOC dataset
    :type dataset_path: str
    :param num_samples: The number of samples in the new dataset
    :type num_samples: int
    :param new_path: Where to save the new dataset
    :type new_path: str

    """
    new_labels_path = os.path.join(new_path, 'Annotations')
    new_images_path = os.path.join(new_path, 'Images')
    os.mkdir(new_path)
    os.mkdir(new_labels_path)
    os.mkdir(new_images_path)

    labels_path = os.path.join(dataset_path, 'Annotations')
    images_path = os.path.join(dataset_path, 'Images')

    # shuffle Annotations/ since make_voc_dataset.py made a csv annotations
    # file only if there was a corresponding image in the new dataset.
    annotations = glob.glob(os.path.join(labels_path, '*.csv'))
    random.shuffle(annotations)

    with open(os.path.join(new_path, 'train.txt'), 'w') as f:
        for annotation in annotations[:num_samples]:
            filename = os.path.splitext(os.path.basename(annotation))[0]
            f.write(filename + '\n')
            shutil.copy2(annotation, new_labels_path)
            image = os.path.join(images_path, filename + '.jpg')
            shutil.copy2(image, new_images_path)


if __name__ == "__main__":

    dataset_path = '../../data/VOC'

    for num_samples in [10, 100]:
        new_path = f'../../data/VOC_{num_samples}'

        assert os.path.exists(dataset_path), "Custom VOC dataset doesn't exist"
        assert not os.path.exists(new_path), f"VOC_{num_samples} exists"

        main(dataset_path, num_samples, new_path)
