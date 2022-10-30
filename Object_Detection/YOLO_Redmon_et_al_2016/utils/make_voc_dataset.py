"""Create a single dataset from the original Pascal VOC 2007 and 2012 datasets.

Created on Fri Oct 28 13:33:06 2022

@author: gonzr


The Pascal VOC 2007 and 2012 datasets are downloaded from:
    https://www.kaggle.com/datasets/vijayabhaskar96/pascal-voc-2007-and-2012

After downloading the datasets, they need to be separated and moved to:
    ../data/
    ├─ VOC2007/
    ├─ VOC2012/

This scipt creates a dataset that has the following directory structure:
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

For simplicity, the datasets are combined and randomly split into a 70/20/10
train, val and test splits.

The labeled dataset is a follows:
    {(image_i, [x_center, y_center, width, height, class_id, class_name]_i)}

"""


import csv
import glob
import numpy as np
import os
import random
import xml.etree.ElementTree as ET

from distutils.dir_util import copy_tree


# https://gist.github.com/anujonthemove/d6d84be473e27057f8d95b89bac50cf9
CLASS_ID__MAP = {
    'aeroplane': 0,
    'bicycle': 1,
    'bird': 2,
    'boat': 3,
    'bottle': 4,
    'bus': 5,
    'car': 6,
    'cat': 7,
    'chair': 8,
    'cow': 9,
    'diningtable': 10,
    'dog': 11,
    'horse': 12,
    'motorbike': 13,
    'person': 14,
    'pottedplant': 15,
    'sheep': 16,
    'sofa': 17,
    'train': 18,
    'tvmonitor': 19
    }


def voc_to_yolo_bbox(labels, im_shape):
    # [x_center, y_center, width, height, class_id]

    labels[:, 2] -= labels[:, 0]        # width
    labels[:, 3] -= labels[:, 1]        # height
    labels[:, 0] += 0.5 * labels[:, 2]  # x_center
    labels[:, 1] += 0.5 * labels[:, 3]  # y_center
    labels[:, 0:-1:2] /= im_shape[0]
    labels[:, 1:-1:2] /= im_shape[1]

    return labels


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
        cls_id = CLASS_ID__MAP[cls_name]

        xml_box = obj.find('bndbox')
        xmin = (float(xml_box.find('xmin').text) - 1)
        ymin = (float(xml_box.find('ymin').text) - 1)
        xmax = (float(xml_box.find('xmax').text) - 1)
        ymax = (float(xml_box.find('ymax').text) - 1)

        labels.append([xmin, ymin, xmax, ymax, cls_id])

    labels = voc_to_yolo_bbox(np.array(labels), (img_w, img_h))

    return labels


def write_csv(labels, dest):
    """

    :param labels: DESCRIPTION
    :type labels: TYPE
    :param dest: DESCRIPTION
    :type dest: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """
    with open(dest, 'w') as f:
        write = csv.writer(f, lineterminator='\n')
        write.writerow(['x_center', 'y_center', 'width', 'height', 'class_id'])
        write.writerows(labels)


def main(new_path):

    path_07 = '../data/VOC2007'
    path_12 = '../data/VOC2012'
    # os.mkdir(new_path)
    # os.mkdir(os.path.join(new_path, 'Annotations'))
    new_images_path = os.path.join(new_path, 'Images')
    new_labels_path = os.path.join(new_path, 'Annotations')

    # for path in [path_07, path_12]:

    #     # combine all images into a single folder
    #     images_path = os.path.join(path, 'JPEGImages')
    #     copy_tree(src=images_path, dst=new_images_path)

    #     # convert to yolo labels and combine in a single folder
    #     labels_path = os.path.join(path, 'Annotations')

    #     for xml_path in glob.glob(f'{labels_path}/*'):

    #         xml_filename = os.path.basename(xml_path)
    #         filename = os.path.splitext(xml_filename)[0]
    #         image_path = os.path.join(new_images_path, filename + '.jpg')

    #         # make sure there is a corresponding image in Images/
    #         if os.path.exists(image_path):
    #             labels = labels_from_xml(xml_path)
    #             dst = os.path.join(new_labels_path, filename + '.csv')
    #             write_csv(labels, dst)

    # make train, val and test splits
    # shuffle Annotations/ since we made a csv annotations file only if there
    # was a corresponding image in the new dataset.
    annotations = glob.glob(os.path.join(new_labels_path, '*.csv'))
    random.shuffle(annotations)

    splits = {"train": 0.7, "val": 0.2}  # 'test' = 1 - (train + val)
    train_idx = int(len(annotations) * splits['train'])
    val_idx = train_idx + (int(len(annotations) * splits['val']))

    for annotation in annotations[:train_idx]:
        print(annotation)

    for annotation in annotations[train_idx:val_idx]:
        print(annotation)

    for annotation in annotations[val_idx:]:
        print(annotation)


if __name__ == "__main__":

    new_path = '../data/VOC/'
    # assert not os.path.exists(new_path)
    main(new_path)
