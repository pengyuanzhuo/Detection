# coding: utf-8
# generate default box

import torch
import numpy as np


class DefaultBox(object):
    def __init__(self, input_size=300,
                 feature_maps=[38, 19, 10, 5, 3, 1],
                 ratios = [3, 2, 1, 1./2, 1./3],
                 min_scale=0.2, max_scale=0.9):
        '''
        input_size: int, input image size, 300 for ssd300
        feature_map: list, feature map size for each output feature map
        ratios: w / h ratios
        min_scale: float
        max_scale: float
        '''
        self.input_size = input_size
        self.feature_maps = feature_maps
        self.ratios = ratios
        self.min_scale = min_scale
        self.max_scale = max_scale

    def forward(self):
        scales = [0.1]
        m = len(self.feature_maps) - 1
        for k in range(1, m + 1):
            scale_k = self.min_scale + (self.max_scale - self.min_scale) * (k - 1) / (m - 1)
            scales += [scale_k]
        print(scales)


if __name__ == '__main__':
    defaultbox = DefaultBox()
    defaultbox.forward()