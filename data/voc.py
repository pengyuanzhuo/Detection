"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
import os
import torch
import torch.utils.data as data
import cv2
import numpy as np
import sys
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

import data.augmentations as aug
import data.collate as collate


class VOCDetection(data.Dataset):
    '''
    voc detection dataset class

    Args:
        root: string, VOCdevkit path
        image_set: list, imageset to use ('train', 'val', 'test')
            [('2007', 'trainval'), ('2012', 'trainval')]
        transform (callable, optional): transformation to perform on the
            input image
        keep_difficult: bool, keep difficult object or not
            default: False
        do_norm: bool, bbox / (w, h) or not
            default: True
    '''
    def __init__(self, root, image_set=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None, keep_difficult=False, do_norm=True):
        super(data.Dataset, self).__init__()
        self.classes = ('background',
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor'
        )
        self.class_index_dict = dict(zip(self.classes, range(len(self.classes))))
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.image_list = []
        self.ann_list = []
        for year, dataset in image_set:
            subdir = os.path.join(self.root, 'VOC' + year)
            ann_dir = os.path.join(subdir, 'Annotations')
            img_dir = os.path.join(subdir, 'JPEGImages')
            for line in open(os.path.join(subdir, 'ImageSets', 'Main', dataset + '.txt')):
                self.image_list.append(os.path.join(img_dir, line.strip() + '.jpg'))
                self.ann_list.append(os.path.join(ann_dir, line.strip() + '.xml'))
        self.keep_difficult = keep_difficult
        self.do_norm = do_norm

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        '''
        return img, target
        target: [[xmin, ymin, xmax, ymax, label],
                 ...,
                 ...]
        '''
        # assert ann and img exist
        # TODO

        img = cv2.imread(self.image_list[index])
        h, w, c = img.shape
        img = img[:, :, ::-1]

        xmlroot = ET.parse(self.ann_list[index]).getroot()

        target = []
        for obj in xmlroot.findall('object'):
            difficult = int(obj.find('difficult').text) == 1
            if difficult and (not self.keep_difficult):
                continue
            classname = obj.find('name').text.lower().strip()
            classlabel = self.class_index_dict[classname]
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text) - 1
            ymin = int(bndbox.find('ymin').text) - 1
            xmax = int(bndbox.find('xmax').text) - 1
            ymax = int(bndbox.find('ymax').text) - 1
            if self.do_norm:
                xmin /= w
                xmax /= w
                ymin /= h
                ymax /= h
            target.append([xmin, ymin, xmax, ymax, classlabel])
        target = np.array(target, dtype=np.float32)

        if self.transform:
            img, bbox, label = self.transform(img, target[:, :4], target[:, 4:])
            target = np.hstack((bbox, label))

        return img, target
