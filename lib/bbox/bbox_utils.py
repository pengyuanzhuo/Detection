import numpy as np
import torch


def xyxy_to_xywh(bboxes):
    '''
    convert [xmin, ymin, xmax, ymax] bboxes to [x, y, w, h]
    Args:
        bboxes: shape=(N, 4)
            vstack of [xmin, ymin, xmax, ymax]
    '''
    xy_c = (bboxes[:, :2] + bboxes[:, 2:]) / 2 # shape=(N, 2)
    wh = (bboxes[:, 2:] - bboxes[:, :2])

    return np.hstack((xy_c, wh))


def xywh_to_xyxy(bboxes):
    '''
    convert [x, y, w, h] => [xmin, ymin, xmax, ymax]
    Args:
        bboxes: shape=(N, 4)
            vstack of [x_c, y_c, w, h]
    '''
    xy_min = bboxes[:, :2] - bboxes[:, 2:] / 2
    xy_max = bboxes[:, :2] + bboxes[:, 2:] / 2

    return np.hstack((xy_min, xy_max)) 


def box_area(bboxes):
    '''
    Computes the area of a set of bounding boxes.
    Args:
        bboxes: vstack of [x1, y1, x2, y2]
            shape=(N, 4)
    Return:
        areas: shape=(N,)
    '''
    w = bboxes[:, 2] - bboxes[:, 0] + 1
    h = bboxes[:, 3] - bboxes[:, 1] + 1
    return w * h


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def box_iou(bboxes1, bboxes2):
    '''
    python baseline
    Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes1: (N, 4)
        boxes2: (K, 4)

    Return:
        iou (N, K)
    '''
    areas1 = box_area(bboxes1) # (N,)
    areas2 = box_area(bboxes2) # (K,)

    xy_i_min = np.maximum(bboxes1[:, :2][:, None, :], bboxes2[:, :2]) # shape=(N, K, 2)
    xy_i_max = np.minimum(bboxes1[:, 2:][:, None, :], bboxes2[:, 2:]) # shape=(N, K, 2)

    wh_i = xy_i_max - xy_i_min + 0
    wh_i = np.maximum(0.0, wh_i) # shape=(N, K, 2)

    inter = wh_i[:, :, 0] * wh_i[:, :, 1] # (N, K)
    union = areas1[:, None] + areas2 - inter # (N, K)

    return inter / union


def encode(anchors, gts, variance=[1.0, 1.0]):
    '''
    Encode the transforms from the priorbox layers into the ground truth boxes
    Args:
        anchors: shape=(N, 4), vstack of [x, y, w, h]
        gts: shape=(N, 4), vstack of [xmin, ymin, xmax, ymax]
        variance: list, len=2. SSD tricks. [1.0, 1.0] to turn off

    Return:
        t: vstack of [tx, ty, tw, th]
    '''
    txy = ((gts[:, :2] + gts[:, 2:]) / 2 - anchors[:, :2]) / anchors[:, 2:] # shape=(N, 2)
    txy /= variance[0]
    twh = np.log((gts[:, 2:] - gts[:, :2]) / anchors[:, 2:])
    twh /= variance[1]

    return np.hstack((txy, twh)) # (N, 4)


def decode(anchors, loc, variance=[1.0, 1.0]):
    '''
    decode the offset to bbox
    Args:
        anchors: shape=(N, 4), vstack of [x, y, w, h]
        loc: shape=(N, 4), vstack of [tx, ty, tw, th]
        variance: list, len=2. SSD trick. [1.0, 1.0] to turn off

    Return:
        bboxes: shape=(N, 4), vstack of [xmin, ymin, xmax, ymax]
    '''
    bboxes_xy = anchors[:, :2] + anchors[:, 2:] * loc[:, :2] * variance[0]
    bboxes_wh = anchors[:, 2:] * np.exp(loc[:, 2:] * variance[1])
    bboxes = np.hstack((bboxes_xy, bboxes_wh))

    return xywh_to_xyxy(bboxes)


if __name__ == "__main__":
    bboxes1 = np.array([[0, 0, 60, 60],
                       [3, 3, 45, 50]], dtype=np.float32)
    bboxes2 = np.array([[10, 10, 25, 25],
                        [22, 12, 100, 100],
                        [32, 12, 50, 80]], dtype=np.float32)
    bboxes1_xywh = xyxy_to_xywh(bboxes1)
    print('xywh => \n', bboxes1_xywh)
    print('xyxy => \n', xywh_to_xyxy(bboxes1_xywh))
    # bboxes1 = torch.from_numpy(bboxes1)
    # bboxes2 = torch.from_numpy(bboxes2)
    print('iou => \n', box_iou(bboxes1, bboxes2))

    anchors = np.array([[10, 10, 25, 25],
                        [22, 12, 100, 100],
                        [32, 12, 50, 80]], dtype=np.float32)
    gts = np.array([[11, 11, 23, 25],
                    [22, 13, 101, 102],
                    [32, 12, 50, 80]], dtype=np.float32)
    # anchors = torch.from_numpy(anchors)
    # gts = torch.from_numpy(gts)
    print('encode => \n', encode(anchors, gts))
    print('decode => \n', decode(anchors, encode(anchors, gts)))
