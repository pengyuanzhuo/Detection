# coding: utf-8

import torch
import torch.nn as nn


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}

extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}


def make_vgg(cfg, batch_norm=False):
    layers = []
    input_channel = 3
    for c in cfg:
        if c == 'M':
            layers += [nn.MaxPool2d(2, stride=2)]
        elif c == 'C':
            layers += [nn.MaxPool2d(2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(input_channel, c, kernel_size=3, stride=1, padding=1)
            input_channel = c
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]

    pool5 = nn.MaxPool2d(3, stride=1, padding=1) # 1024*19*19
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6) # 1024*19*19
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1) # 1024*19*19
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    return layers


def make_extras():
    extra1_1 = nn.Conv2d(1024, 256, kernel_size=1) # (256, 19, 19)
    extra1_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1) # (512, 10, 10)

    extra2_1 = nn.Conv2d(512, 128, kernel_size=1) # (128, 10, 10)
    extra2_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # (256, 5, 5)

    extra3_1 = nn.Conv2d(256, 128, kernel_size=1) # (128, 5, 5)
    extra3_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0) # (256, 3, 3)

    extra4_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # (128, 3, 3)
    extra4_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0) # (256, 1, 1)

    return [extra1_1, extra1_2, extra2_1, extra2_2, extra3_1, extra3_2, extra4_1, extra4_2]

