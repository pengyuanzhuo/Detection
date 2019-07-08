import torch
import torch.optim as optim

from data import VOCDetection, detection_collate
from data import Resize, Compose, ToTensor
from models.ssd import build_ssd
import lib.bbox as bbox
import models.multibox_loss as multibox_loss
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
    losses = AverageMeter()

    tic = time.time()
    for imgs, targets in train_loader:
        # imgs.shape = (b, 3, h, w)
        # targets = [target, target, ...], len(targets) = b, target.shape = (n_obj, 5)
        data_time.update((time.time() - tic) * 1000)
        if DEBUG:
            # show train sample for debug
            img, target = imgs[0], targets[0]
            img = img.numpy().transpose((1, 2, 0)).copy().astype(np.uint8)
            target = target.numpy()[:, :4].copy()
            for box in target:
                box *= args.input_size
                cv2.rectangle(img,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    (0, 255, 0), thickness=1
                )
            plt.imshow(img)
            plt.savefig('./demo.jpg', dpi=600)

        # default_box, shape=(n_anchors, 4), vstack of [cx, cy, scale_w, scale_h]
        # conf, shape=(b, n_anchors, n_classes)
        # loc, shape=(b, n_anchors, 4)
        default_box, loc, conf = model(imgs)
        n_anchors, _ = default_box.shape

        if DEBUG:
            print('=> model output:')
            print('default box => ', default_box.shape)
            print('conf =>', conf.shape)
            print('loc =>', loc.shape)

        batch_loc_target = np.zeros((args.batch_size, n_anchors, 4))
        batch_conf_target = np.zeros((args.batch_size, n_anchors))
        for i, target in enumerate(targets):
            target = target.numpy() # torch.Tensor => numpy.
            loc_target, conf_target, matches = bbox.match(default_box, target, args.threshold, args.variances)
            # loc_target.shape = (n_anchors, 4)
            # conf_target.shape = (n_anchors,)
            batch_loc_target[i] = loc_target
            batch_conf_target[i] = conf_target

        if DEBUG:
            print('=> positive default_box: ')
            pos_default_box = default_box[batch_conf_target[0] > 0]
            pos_default_box = bbox.xywh_to_xyxy(pos_default_box)
            img = imgs[0].numpy().transpose((1, 2, 0)).copy().astype(np.uint8)
            for box in pos_default_box:
                box *= args.input_size
                cv2.rectangle(img,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    (0, 255, 0), thickness=1
                )
            for box in matches[batch_conf_target[0] > 0]:
                box *= args.input_size
                cv2.rectangle(img,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    (255, 0, 0), thickness=1
                )
            plt.imshow(img)
            plt.savefig('./sample.jpg', dpi=600)

        batch_loc_target = torch.from_numpy(batch_loc_target).float()
        batch_conf_target = torch.from_numpy(batch_conf_target).long()
        loss = criterion(loc, conf, batch_loc_target, batch_conf_target)
        if DEBUG:
            print('=> MultiboxLoss: ')
            print('loc loss:', loss.shape)
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
    criterion = multibox_loss.MultiboxLoss(args.num_classes)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay = args.weight_decay)

    args.start_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, args)


if __name__ == '__main__':
    main(cfg)
