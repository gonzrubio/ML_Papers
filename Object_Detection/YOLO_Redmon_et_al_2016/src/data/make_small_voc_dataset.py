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

import os


def main(new_path):
    pass


if __name__ == "__main__":

    dataset_path = '../../data/VOC'
    num_samples = 10
    new_path = f'../../data/VOC_{num_samples}'

    assert os.path.exists(dataset_path), "Custom VOC dataset doesn't exist"
    assert not os.path.exists(new_path), f"VOC_{num_samples} already exists"

    main(dataset_path, num_samples, new_path)
