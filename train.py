import torch
import torch.optim as optim

from data import VOCDetection, detection_collate
from data import Resize, Compose, ToTensor, Norm
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
import os

DEBUG = False


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


def set_learning_rate(optimizer, epoch, iter_size, iter_num, args):
    lr_step = args.lr_step
    current_iter = epoch * iter_size + iter_num
    current_lr = args.lr
    if current_iter in lr_step:
        current_lr =  args.lr * (0.1 ** (lr_step.index(current_iter) + 1))
        args.lr = current_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    return current_lr


def train(train_loader, model, criterion, optimizer, epoch, args):
    model.train()

    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_conf = AverageMeter()
    losses_loc = AverageMeter()

    tic = time.time()
    for i, (imgs, targets) in enumerate(train_loader):
        # imgs.shape = (b, 3, h, w)
        # targets = [target, target, ...], len(targets) = b, target.shape = (n_obj, 5)
        data_time.update((time.time() - tic) * 1000)

        lr = set_learning_rate(optimizer, epoch, len(train_loader), i+1, args)
        batch_size = imgs.size(0)
        if args.gpus:
            imgs = imgs.cuda(non_blocking=True)

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

        batch_loc_target = np.zeros((batch_size, n_anchors, 4))
        batch_conf_target = np.zeros((batch_size, n_anchors))
        for j, target in enumerate(targets):
            target = target.numpy() # torch.Tensor => numpy.
            loc_target, conf_target, matches = bbox.match(default_box, target, args.threshold, args.variances)
            # loc_target.shape = (n_anchors, 4)
            # conf_target.shape = (n_anchors,)
            batch_loc_target[j] = loc_target
            batch_conf_target[j] = conf_target

        if DEBUG:
            print('=> positive default_box: ')
            pos_default_box = default_box[batch_conf_target[0] > 0]
            pos_default_box = bbox.xywh_to_xyxy(pos_default_box)
            img = imgs[0].numpy().transpose((1, 2, 0)).copy()
            img += np.array([123, 117, 104])
            img = img.astype(np.uint8)
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
            plt.savefig('./samples/sample_{}.jpg'.format(i), dpi=600)

        batch_loc_target = torch.from_numpy(batch_loc_target).float()
        batch_conf_target = torch.from_numpy(batch_conf_target).long()
        if args.gpus:
            batch_conf_target = batch_conf_target.cuda()
            batch_loc_target = batch_loc_target.cuda()
        loss_conf, loss_loc, loss_merge = criterion(loc, conf, batch_loc_target, batch_conf_target)
        losses.update(loss_merge.item(), batch_size)
        losses_conf.update(loss_conf.item(), batch_size)
        losses_loc.update(loss_loc.item(), batch_size)

        if DEBUG:
            print('=> MultiboxLoss: ')
            print('conf loss:', loss_conf)
            print('loc loss:', loss_loc)
            print('loss mergs:', loss_merge)
            break

        optimizer.zero_grad()
        loss_merge.backward()
        optimizer.step()

        batch_time.update(time.time() - tic)
        tic = time.time()

        if (i+1) % args.print_freq == 0:
            print(time.strftime('%m/%d %H:%M:%S', time.localtime()), end='\t')
            print('=> Train Epoch: [{0}][{1}/{2}]\n'
                  '\tBatch Time {batch_time.val:.3f}({batch_time.avg:.3f})\n'
                  '\tLoss {loss.val:.4f}({loss.avg:.4f})\n'
                  '\tCls Loss {loss_conf.val:.4f}({loss_conf.avg:.4f})\n'
                  '\tLoc Loss {loss_loc.val:.4f}({loss_loc.avg:.4f})\n'
                  '\tLr {lr:.6f}'
                  .format(epoch, i+1, len(train_loader),
                          batch_time=batch_time,
                          loss=losses,
                          loss_conf=losses_conf,
                          loss_loc=losses_loc,
                          lr=lr), flush=True)


def main(args):
    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        print('Using {} GPUs'.format(args.gpus))

    train_transform = Compose([Resize(args.input_size),
                               ToTensor(),
                               Norm(mean=(123, 117, 104))])
    trainset = VOCDetection(args.data_root, args.train_set,
                            transform=train_transform,
                            do_norm=True)
    train_loader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=args.batch_size,
                                               num_workers=args.workers, collate_fn=detection_collate)

    model = build_ssd(cfg)
    if not args.checkpoint and args.pretrain:
        print('load pretrain model: {}'.format(args.pretrain))
        model.load_weight(args.pretrain)
    if args.gpus:
        model = torch.nn.DataParallel(model).cuda()
    criterion = multibox_loss.MultiboxLoss(args.num_classes, args.neg_pos_ratio)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay = args.weight_decay)
    args.start_epoch = 0

    if args.checkpoint:
        print('=> loading checkpoint from {}...'.format(args.checkpoint))
        state = torch.load(args.checkpoint)
        args.start_epoch = state['epoch']
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])

    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, args)

        state = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        # save checkpoint
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(args.checkpoint_dir,
                                       'checkpoint_epoch_{:04d}.pth.tar'.format(state['epoch']))
        torch.save(state, checkpoint_file)


if __name__ == '__main__':
    main(cfg)
