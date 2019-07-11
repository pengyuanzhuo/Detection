# coding: utf-8

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import models.ssd as ssd
import models.detection as detection
from config import Config as cfg

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

CLASSES = ('background',
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor'
)

def parse():
    args = argparse.ArgumentParser('ssd demo')
    args.add_argument('img', type=str, help='img path')
    args.add_argument('model', type=str, help='model path')
    args.add_argument('--topk', '-k', type=int, default=10,
        help='top k bbox, default=10')

    return args.parse_args()


def preprocess(image):
    x = cv2.resize(image, (300, 300)).astype(np.float32)
    x -= (104.0, 117.0, 123.0) # bgr
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = x.unsqueeze(0)

    return x


def build_model(checkpoint, device):
    model = ssd.build_ssd(cfg)
    state = torch.load(checkpoint, map_location=device)
    state_dict = dict()
    for k, v in state['model'].items():
        state_dict[k.replace('module.','')] = v
    model.load_state_dict(state_dict)

    return model


def main(args):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    image = cv2.imread(args.img)
    inputs = preprocess(image) # (1, c, h, w)

    model = build_model(args.model, device).to(device)

    with torch.no_grad():
        inputs = inputs.to(device)

        default_box, loc, conf = model(inputs)
        conf = F.softmax(conf, dim=-1)
        outputs = detection.detection(default_box, conf, loc,
                                      conf_threshold=0.5,
                                      nms_threshold=0.2,
                                      topk=args.topk,
                                      variance=cfg.variance) # shape=(b, num_classes, topk, 5)

        scale = torch.Tensor([image.shape[1::-1]]).repeat(1, 2).squeeze() # [w, h, w, h]
        for i in range(outputs.size(1)):
            j = 0 # 每一类的bbox序号
            while outputs[0, i, j, -1] > 0:
                score = outputs[0, i, j, -1]
                label = i
                # label_name = TODO
                bbox = (outputs[0, i, j, :-1]*scale).cpu().numpy() # (xmin, ymin, xmax, ymax)
                cv2.rectangle(image,
                              (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])),
                              COLORS[i % 3], 2)
                cv2.putText(image, CLASSES[label], (int(bbox[0]), int(bbox[1])),
                            FONT, 2, (255, 255, 255), 2, cv2.LINE_AA)
                j += 1

    cv2.imwrite('./demo.jpg', image)


if __name__ == '__main__':
    args = parse()
    main(args)
