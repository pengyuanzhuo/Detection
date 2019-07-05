import torch
import torch.optim as optim

from data import VOCDetection, detection_collate
from data import Resize, Compose, ToTensor
from models.ssd import build_ssd
from config import Config as cfg

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

DEBUG = True


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, model, criterion, optimizer, epoch, args):
    model.train()

    data_time = AverageMeter()
    batch_time = AverageMeter()
    loss = AverageMeter()

    tic = time.time()
    for imgs, targets in train_loader:
        data_time.update((time.time() - tic) * 1000)
        if DEBUG:
            # show train sample for debug
            img, target = imgs[0], targets[0]
            img = img.numpy().transpose((1, 2, 0)).copy().astype(np.uint8)
            target = target.numpy()[:, :4]
            for box in target:
                box *= args.input_size
                cv2.rectangle(img,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    (0, 255, 0), thickness=1
                )
            plt.imshow(img)
            plt.savefig('./demo.jpg', dpi=600)

        default_box, conf, loc = model(imgs)

        if DEBUG:
            print('default box => ', default_box.shape)
            print('conf =>', conf.shape)
            print('loc =>', loc.shape)
            break


def main(args):
    train_transform = Compose([Resize(args.input_size),
                               ToTensor()])
    trainset = VOCDetection(args.data_root, args.train_set,
                            transform=train_transform,
                            do_norm=True)
    train_loader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=args.batch_size,
                                               num_workers=args.workers, collate_fn=detection_collate)

    model = build_ssd(cfg)
    criterion = None
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay = args.weight_decay)

    args.start_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, args)


if __name__ == '__main__':
    main(cfg)
