# coding: utf-8

from lib.bbox.bbox_utils import box_iou, encode, decode, xywh_to_xyxy
import numpy as np


def match(anchors, gts, threshold=0.5, variances=[1.0, 1.0]):
    '''
    Match each prior box with the ground truth box.
    1) 
    Args:
        anchors: shape=(N, 4), vstack of [x_c, y_c, w, h]
        gts: shape=(K, 5), vstack of [xmin, ymin, xmax, ymax, label_index]. N > K
        threshold: (float) The overlap threshold used when mathing boxes
        variances: ssd tricks. [1.0, 1.0] to turn off

    Return:
        transform_target: shape=(N, 4)
        conf_target: shape=(N,)
    '''
    gts_bbox = gts[:, :4] # (K, 4)
    gts_label = gts[:, 4] # (K,)
    iou_mat = box_iou(gts_bbox, xywh_to_xyxy(anchors)) # shape=(K, N)

    # 1) match gt to anchors
    # 对每个gt, 找到最匹配的default box
    n_gts, n_anchors = iou_mat.shape
    best_anchor_ids = np.argmax(iou_mat, axis=1) # shape=(n_gt,) anchor index
    best_anchor_iou = iou_mat[np.arange(n_gts), best_anchor_ids] # shape=(n_gt,) iou(gt, anchor)

    # 2) match anchors to gt
    # 对每个default box, 找到最匹配的gt
    best_gt_ids = np.argmax(iou_mat, axis=0) # shape=(n_anchors,) 为每一个anchor找到最匹配的gt index
    best_gt_iou = iou_mat[best_gt_ids, np.arange(n_anchors)] # shape=(n_anchors,)

    # 对于已经被gt匹配的anchor, 其对应的gt显然应该由step1指定
    for i in range(best_anchor_ids.shape[0]):
        # 其中 i 是gt的index
        best_gt_ids[best_anchor_ids[i]] = i

    # 还要保证与gt匹配的anchor的iou足够高, 不被设为负样本
    best_gt_iou[best_anchor_ids] = 2
    print(best_gt_iou)

    matches = gts_bbox[best_gt_ids] # 与anchors一一对应的gt box, shape=(N, 4)
    conf_target = gts_label[best_gt_ids] # 与 anchors一一对应的gt label, shape=(N,)
    conf_target[best_gt_iou < threshold] = 0

    transform_target = encode(anchors, matches, variances)

    return transform_target, conf_target


if __name__ == '__main__':
    iou_mat = np.array([[1, 3, 2],
                        [2, 1, 8]])
    n_gts, n_anchors = iou_mat.shape
    best_gt_ids = np.argmax(iou_mat, axis=0) # shape=(n_anchors,) 为每一个anchor找到最匹配的gt index
    best_gt_iou = iou_mat[best_gt_ids, np.arange(n_anchors)]
    print(best_gt_ids)
    print(best_gt_iou)
