import torch
from data import Resize, VOCDetection, detection_collate
from models.ssd import build_ssd
from config import Config as cfg

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2

if __name__ == '__main__':
    dataset = VOCDetection('/Users/qiniu/data/VOCdevkit', [('2007', 'trainval')],
        transform=Resize(300),
        do_norm=True)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=1, collate_fn=detection_collate)
    for imgs, targets in dataloader:
        img, target = imgs[0], targets[0]
        img = img.numpy().transpose((1, 2, 0)).copy()
        target = target.numpy()[:, :4]
        for box in target:
            box *= 300
            print(box)
            cv2.rectangle(img,
                      (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])),
                      (0, 255, 0), thickness=1)
        plt.imshow(img)
        plt.savefig('./demo.jpg')
        break

    '''
    model = build_ssd(cfg)
    inputs = torch.zeros((2, 3, 300, 300))
    # print(inputs.shape)
    default_box, conf, loc = model(inputs)
    # print(default_box.shape)
    # print(conf.shape)
    '''
