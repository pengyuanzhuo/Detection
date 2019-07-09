import torch

import lib.bbox.bbox_utils as bbox_utils


def detection(default_box, conf, loc, conf_threshold=0.8, nms_threshold=0.2, topk=200, variance=[1.0, 1.0]):
    '''
    detection
    Args:
    ----
        default_box: shape=(n_anchors, 4), vstack of [cx, cy, scale_w, scale_h]
        conf: shape=(b, n_anchors, n_classes)
        loc: shape=(b, n_anchors, 4)
        conf_threshold: float, default=0.8
        nms_threshold: float, default=0.2
        topk: TODO
    Return:
    ------
        bbox: shape=
    '''
    batch_size = conf.size(0)
    n_anchors = default_box.size(0)

    for i in range(batch_size):
        loc_i = loc[i]
        decode_boxes = bbox_utils.decode(default_box.numpy(), loc_i.numpy(), variance) # shape=(n_anchors, 4)