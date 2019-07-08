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
    def __init__(self, n_classes):
        super(MultiboxLoss, self).__init__()
        self.n_classes = n_classes

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
        loss_loc = F.smooth_l1_loss(loc_pos, loc_target_pos, reduction='none') # (n_pos, 4)


        # Classification Loss
        batch_conf = conf.view(-1, self.n_classes) # (b*n_anchors, n_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_target.view(-1, 1)) # shape=(b*n_anchors, 1)
        print(loss_c[pos].shape)


        return loss_loc