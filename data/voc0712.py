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


class VOCDetection(data.Dataset):
    '''
    voc detection dataset class

    Args:
        root: string, VOCdevkit path
        image_set: list, imageset to use ('train', 'val', 'test')
            [('2007', 'trainval'), ('2012', 'trainval')]
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform
            on the target `annotation`
    '''
    def __init__(self, root, image_set=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None, target_transform=None):
        super(data.Dataset, self).__init__()
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        self.image_list = []
        self.ann_list = []
        for year, dataset in image_set:
            subdir = os.path.join(self.root, 'VOC' + year)
            ann_dir = os.path.join(subdir, 'Annotation')
            img_dir = os.path.join(subdir, 'JPEGImages')
            for line in open(os.path.join(subdir, 'ImageSets', 'Main', name + '.txt')):
                self.image_list.append(os.path.join(img_dir, line.strip() + '.jpg'))
                self.ann_list.append(os.path.join(ann_dir, line.strip() + '.txt'))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        # assert ann and img exist
        # TODO

        img = cv2.imread(self.image_list[index])
        h, w, c = img.shape
        target = ET.parse(self.ann_list[index]).getroot()
