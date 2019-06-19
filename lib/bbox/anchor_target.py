from bbox_utils import box_iou, encode, decode, xywh_to_xyxy
import numpy as np


def match(anchors, gts, threshold=0.5, variances=[1.0, 1.0]):
    '''
    Match each prior box with the ground truth box.
    1) 
    Args:
        anchors: shape=(N, 4), vstack of [x_c, y_c, w, h]
        gts: shape=(K, 5), vstack of [xmin, ymin, xmax, ymax, label_index]
        threshold: (float) The overlap threshold used when mathing boxes
        variances: 

    Return:
        anchor_targets: shape=(N, 5)
            vstack of [tx, ty, tw, th, label_index]
    '''
    iou_mat = box_iou(gts[:, :4], xywh_to_xyxy(anchors)) # shape=(K, N)

    # 1) match gt to anchors
    n_gts, n_anchors = iou_mat.shape
    best_anchor_ids = np.argmax(iou_mat, axis=1) # shape=(n_gt,) anchor index
    best_anchor_iou = iou_mat[np.arange(n_gts), best_anchor_ids] # shape=(n_gt,) iou(gt, anchor)

    # 2) match anchors to gt
    best_gt_ids = np.argmax(iou_mat, axis=0) # shape=(n_anchors,) 为每一个anchor找到最匹配的gt index
    best_gt_iou = iou_mat[best_gt_ids, np.arange(n_anchors)]

    



if __name__ == '__main__':
    iou_mat = np.array([[1, 3, 2],
                        [2, 1, 8]])
    n_gts, n_anchors = iou_mat.shape
    best_gt_ids = np.argmax(iou_mat, axis=0) # shape=(n_anchors,) 为每一个anchor找到最匹配的gt index
    best_gt_iou = iou_mat[best_gt_ids, np.arange(n_anchors)]
    print(best_gt_ids)
    print(best_gt_iou)
