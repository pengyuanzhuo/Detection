# coding: utf-8
# generate default box

import torch
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import cv2


class DefaultBox(object):
    def __init__(self, input_size=300,
                 feature_maps=[38, 19, 10, 5, 3, 1],
                 # feature_maps=[3, 3, 1],
                 # ratios=[[2], [2]],
                 ratios = [[2], [3, 2], [3, 2], [3, 2], [2], [2]],
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
        '''
        return: 2d array, shape=(N, 4)
            vstack of [cx, cy, scale_w, scale_h]
            [cx, cy, scale_w, scale_h] * img_size => default box in img
        '''
        scales = [0.1]
        m = len(self.feature_maps) - 1
        for k in range(1, m + 1):
            # m + 2 for aux scale
            scale_k = self.min_scale + (self.max_scale - self.min_scale) * (k - 1) / ((m - 1) + 1e-9)
            scales += [scale_k]
        scales += [self.max_scale + 0.5 * (1 - self.max_scale)]

        default_box = []
        for i, fm_size in enumerate(self.feature_maps):
            a, b = np.meshgrid(np.arange(fm_size), np.arange(fm_size))
            fm_xy = np.hstack((a.reshape((-1, 1)), b.reshape((-1, 1))))
            # 遍历当前feature map的每一个cell, 计算default box的中心
            for x_i, y_i in fm_xy:
                cx = (y_i + 0.5) / fm_size
                cy = (x_i + 0.5) / fm_size
                # 在每个cell计算不同ratio的default box
                # ratio = 1时, 生成2个scale的default box
                default_box += [cx, cy, scales[i], scales[i]]
                default_box += [cx, cy, np.sqrt(scales[i] * scales[i + 1]), np.sqrt(scales[i] * scales[i + 1])]
                for ratio in self.ratios[i]:
                    default_box += [cx, cy, np.sqrt(ratio) * scales[i], scales[i] / np.sqrt(ratio)]
                    default_box += [cx, cy, scales[i] / np.sqrt(ratio), scales[i] * np.sqrt(ratio)]

        default_box = np.array(default_box, dtype=np.float32).reshape((-1, 4))
        default_box = np.clip(default_box, 0, 1)

        return default_box

    def _xywh_to_xyxy(self, bboxes):
        '''
        convert [x, y, w, h] => [xmin, ymin, xmax, ymax]
        Args:
            bboxes: shape=(N, 4)
                vstack of [x_c, y_c, w, h]
        '''
        xy_min = bboxes[:, :2] - bboxes[:, 2:] / 2
        xy_max = bboxes[:, :2] + bboxes[:, 2:] / 2

        return np.hstack((xy_min, xy_max))

    def draw_default_box(self):
        background = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        background.fill(255)

        default_boxs = self.forward()
        default_boxs = default_boxs * self.input_size
        default_boxs = self._xywh_to_xyxy(default_boxs)
        print(default_boxs.shape)
        for default_box in default_boxs:
            cv2.rectangle(background,
                          (default_box[0], default_box[1]),
                          (default_box[2], default_box[3]),
                          (0, 255, 0), thickness=1)

        plt.imshow(background)
        plt.savefig('./defaultbox_demo.jpg')


if __name__ == '__main__':
    defaultbox = DefaultBox()
    defaultbox.draw_default_box()
