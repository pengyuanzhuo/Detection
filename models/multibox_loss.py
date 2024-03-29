import torch
import torch.nn as nn
import torch.nn.functional as F


def log_sum_exp(x):
    '''
    log(exp(x)) but num stable
    args:
        x: x.shape=(n_anchors, n_classes)
    '''
    x_max = x.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


class MultiboxLoss(nn.Module):
    def __init__(self, n_classes, neg_pos_ratio):
        super(MultiboxLoss, self).__init__()
        self.n_classes = n_classes
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, loc, conf, loc_target, conf_target):
        '''
        MultiboxLoss
        args:
        ----
            loc: loc transform from model, shape=(b, n_anchors, 4)
            conf: class conf from model, shape=(b, n_anchors, n_classes)
            loc_target: loc transform from 'match', shape=(b, n_anchors, 4)
            conf_target: conf from 'match', shape=(b, n_anchors)
        '''
        # Localization Loss (Smooth L1)
        # find positive sample
        pos = conf_target > 0 # shape=(b, n_anchors)
        n_pos = pos.sum(dim=1, keepdim=True) # (b, 1)

        pos_mask = pos.unsqueeze(pos.dim()).expand_as(loc) # (b, n_anchor) => (b, n_anchor, 4)
        loc_pos = loc[pos_mask].view(-1, 4) # (4*n_pos, ) => (n_pos, 4)
        loc_target_pos = loc_target[pos_mask].view(-1, 4) # (n_pos, 4)
        loss_loc = F.smooth_l1_loss(loc_pos, loc_target_pos, reduction='sum') # (n_pos, 4) if reduction='none'

        # Classification Loss
        batch_conf = conf.view(-1, self.n_classes) # (b*n_anchors, n_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_target.view(-1, 1)) # shape=(b*n_anchors, 1)
        loss_c[pos.reshape((-1, 1))] = 0 # negative sample only
        # sort by loss_c
        batch_size = conf.size(0)
        loss_c = loss_c.reshape(batch_size, -1) # shape=(b, n_anchors). 每一行表示一张图的anchor loss
        # 每一张图片的anchor需要分别处理
        # 两次排序, 生成的是每个元素的排位, 例如
        # 原序列x = [0, 3, 1]
        # x.sort 降序 => [1, 2, 0]
        # 再排序 => [2, 0, 1]
        # 原数组0排第2位, 原数组3排第0位, 原数组1排第1位 
        _, loss_indices = loss_c.sort(dim=1, descending=True) # shape=(b, n_anchors)
        _, loss_rank = loss_indices.sort(dim=1) # shape=(b, n_anchors)
        num_pos = pos.long().sum(dim=1, keepdim=True) # shape=(b, 1). batch内每张图的pos样本数
        num_neg = torch.clamp(self.neg_pos_ratio*num_pos, max=pos.size(1)-1) # shape=(b, 1)
        # 选loss高的样本前num_neg个样本作为负样本
        neg = loss_rank < num_neg # shape=(b, n_anchors)

        pos_mask = pos.unsqueeze(2).expand_as(conf) # shape=(b, n_anchor, n_classes)
        neg_mask = neg.unsqueeze(2).expand_as(conf) # shape=(b, n_anchor, n_classes)

        conf_mining = conf[pos_mask+neg_mask].view(-1, self.n_classes) # shape=(n_mining, n_classes)
        conf_target_mining = conf_target[pos+neg] # shape=(n_mining,)
        loss_conf = F.cross_entropy(conf_mining, conf_target_mining, reduction='sum') # shape=(n_mining,)

        N = n_pos.sum()
        loss_conf /= N
        loss_loc /= N

        return loss_conf, loss_loc, loss_loc+loss_conf
