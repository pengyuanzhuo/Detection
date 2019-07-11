import torch

import lib.bbox.bbox_utils as bbox_utils
import lib.nms.py_cpu_nms as nms


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
        topk: 
    Return:
    ------
        bbox: shape=
    '''
    batch_size = conf.size(0)
    n_classes = conf.size(2)
    n_anchors = default_box.shape[0]

    output = torch.zeros(batch_size, n_classes, topk, 5)
    for i in range(batch_size):
        loc_i = loc[i] # shape=(n_anchors, 4)
        decode_boxes = bbox_utils.decode(default_box, loc_i.cpu().numpy(), variance) # shape=(n_anchors, 4)

        conf_i = conf[i] # shape=(n_anchors, n_classes), 第i张图的conf
        # 对每一类分别执行nms
        for c in range(1, n_classes):
            conf_i_c = conf_i[:, c] # 第i张图的第c个类别, 各anchor的score. shape=(n_anchors,)
            conf_i_c_mask = conf_i_c > conf_threshold # dtype=torch.uint8, shape=(n_anchors,)
            conf_i_scores = conf_i_c[conf_i_c_mask] # shape=(n_score_greater_threshold,), > conf_threshold的score
            if conf_i_scores.size(0) == 0:
                continue
            decode_boxes_c = torch.from_numpy(decode_boxes[conf_i_c_mask.cpu().numpy()]) # shape=(n_score_greter_threshold, 4) > conf_threshold的bbox

            # convert to numpy, nms
            conf_i_scores = conf_i_scores.unsqueeze(-1) # (n_score_greater_threshold,) => (n_score_greater_threshold, 1)
            bbox_c = torch.cat((decode_boxes_c, conf_i_scores.cpu()), 1) # vstack of [x1, y1, x2, y2, score]
            keep = nms.py_cpu_nms(bbox_c.numpy(), nms_threshold) # python list
            bbox_c = bbox_c[keep] # shape=(n_bbox_nms, 5)
            num_bbox_c = bbox_c.size(0)

            # fill to output
            output[i, c, :num_bbox_c] = bbox_c

    # 考虑所有类别, 保留topk
    flt = output.contiguous().view(batch_size, -1, 5) # shape=(b, all_class_after_nms, 5)
    _, index = flt[:, :, -1].sort(1, descending=True) # shape=(b, all_class_after_nms)
    _, rank = index.sort(1) # shape=(b, all_class_after_nms)
    flt[(rank > topk).unsqueeze(-1).expand_as(flt)].fill_(0)

    return output
