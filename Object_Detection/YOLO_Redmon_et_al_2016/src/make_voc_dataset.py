"""Create a single dataset from the original Pascal VOC 2007 and 2012 datasets.

Created on Fri Oct 28 13:33:06 2022

@author: gonzr


This script downloads the datasets and creates a single dataset for object
detection. The directory of the custom dataset has the following structure:
    data/VOC/
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

For simplicity, the datasets are combined and randomly split (70/20/10) into
train, val and test splits. All of the images are resized to 448x448 and
standardized using the mean and the standard deviation of ImageNet. The center
coordinates, width and height of the bounding boxes are normalized to be
relative to the resized images (between zero and one).

The labeled dataset is a follows:
    {(image_i, [x_center, y_center, width, height, class_id, class_name]_i)}

The Pascal VOC 2007 and 2012 datasets are downloaded from:
    https://www.kaggle.com/datasets/vijayabhaskar96/pascal-voc-2007-and-2012

The class labels follow:
    https://gist.github.com/anujonthemove/d6d84be473e27057f8d95b89bac50cf9

The color map for bounding boxes of detected objects is from:
    https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/

"""

import csv
import glob
import numpy as np
import os
import random
import xml.etree.ElementTree as ET

from distutils.dir_util import copy_tree
from PIL import Image
from utils.bounding_boxes import voc_to_yolo_bbox

CLASS_ID_MAP = {
    'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5,
    'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
    'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16,
    'sofa': 17, 'train': 18, 'tvmonitor': 19
    }

ID_CLASS_MAP = {
    0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle', 5: 'bus',
    6: 'car', 7: 'cat', 8: 'chair', 9: 'cow', 10: 'diningtable', 11: 'dog',
    12: 'horse', 13: 'motorbike', 14: 'person', 15: 'pottedplant', 16: 'sheep',
    17: 'sofa', 18: 'train', 19: 'tvmonitor'
    }

ID_COLOR_MAP = {
    0: '#e6194b', 1: '#3cb44b', 2: '#ffe119', 3: '#0082c8', 4: '#f58231',
    5: '#911eb4', 6: '#46f0f0', 7: '#f032e6', 8: '#d2f53c', 9: '#fabebe',
    10: '#008080', 11: '#000080', 12: '#aa6e28', 13: '#fffac8', 14: '#800000',
    15: '#aaffc3', 16: '#808000', 17: '#ffd8b1', 18: '#e6beff', 19: '#808080'
    }


def labels_from_xml(xml_path):
    """Parse the xml file and return the labels.

    :param xml_path: path to the xml file
    :type xml_path: str
    :return: labels
    :rtype: numpy.ndarray

    """
    labels = []
    root = ET.parse(xml_path).getroot()
    img_w = int(root.find("size").find("width").text)
    img_h = int(root.find("size").find("height").text)

    for obj in root.iter('object'):

        cls_name = obj.find('name').text.strip().lower()
        cls_id = CLASS_ID_MAP[cls_name]

        xml_box = obj.find('bndbox')
        xmin = (float(xml_box.find('xmin').text) - 1)
        ymin = (float(xml_box.find('ymin').text) - 1)
        xmax = (float(xml_box.find('xmax').text) - 1)
        ymax = (float(xml_box.find('ymax').text) - 1)

        labels.append([xmin, ymin, xmax, ymax, cls_id])

    labels = voc_to_yolo_bbox(np.array(labels), (img_w, img_h))

    return labels


def write_csv(labels, dest):
    """Create a csv file from 'labels'.

    :param labels: Bounding box coordinates and class number
    :type labels: numpy.ndarray
    :param dest: Full path to the csv
    :type dest: str

    """
    with open(dest, 'w') as f:
        write = csv.writer(f, lineterminator='\n')
        write.writerow(['x_center', 'y_center', 'width', 'height', 'class_id'])
        write.writerows(labels)


def main(new_path):
    """Create a single dataset from the Pascal VOC 2007 and 2012 datasets.

    :param new_path: DESCRIPTION
    :type new_path: TYPE

    """
    os.mkdir(new_path)
    os.mkdir(os.path.join(new_path, 'Annotations'))

    new_images_path = os.path.join(new_path, 'Images')
    new_labels_path = os.path.join(new_path, 'Annotations')

    path_07 = './VOC2007'
    path_12 = './VOC2012'

    for path in [path_07, path_12]:

        # combine all images into a single folder
        images_path = os.path.join(path, 'JPEGImages')
        copy_tree(src=images_path, dst=new_images_path)

        # convert to yolo labels and combine in a single folder
        labels_path = os.path.join(path, 'Annotations')

        # make sure there is a corresponding image in Images/ for each
        # annotation and resize the image to 448x448
        for xml_path in glob.glob(f'{labels_path}/*'):

            xml_filename = os.path.basename(xml_path)
            filename = os.path.splitext(xml_filename)[0]
            image_path = os.path.join(new_images_path, filename + '.jpg')

            if os.path.exists(image_path):
                image = Image.open(image_path)
                image = image.resize((448, 448))
                image.save(image_path)
                labels = labels_from_xml(xml_path)
                dst = os.path.join(new_labels_path, filename + '.csv')
                write_csv(labels, dst)

    # make train, val and test splits
    # shuffle Annotations/ since we made a csv annotations file only if there
    # was a corresponding image in the new dataset.
    annotations = glob.glob(os.path.join(new_labels_path, '*.csv'))
    random.shuffle(annotations)

    splits = {"train": 0.7, "val": 0.2}  # 'test' = 1 - (train + val)
    train_idx = int(len(annotations) * splits['train'])
    val_idx = train_idx + (int(len(annotations) * splits['val']))

    with open(os.path.join(new_path, 'train.txt'), 'w') as f:
        for annotation in annotations[:train_idx]:
            filename = os.path.splitext(os.path.basename(annotation))[0]
            f.write(filename + '\n')

    with open(os.path.join(new_path, 'val.txt'), 'w') as f:
        for annotation in annotations[train_idx:val_idx]:
            filename = os.path.splitext(os.path.basename(annotation))[0]
            f.write(filename + '\n')

    with open(os.path.join(new_path, 'test.txt'), 'w') as f:
        for annotation in annotations[val_idx:]:
            filename = os.path.splitext(os.path.basename(annotation))[0]
            f.write(filename + '\n')


if __name__ == "__main__":

    new_path = '../data/VOC'
    assert not os.path.exists(new_path), "Custom VOC dataset already exists"
    main(new_path)
