import numpy as np


def box_area(bboxes):
    '''
    Computes the area of a set of bounding boxes.
    Args:
        bboxes: vstack of [x1, y1, x2, y2]
            shape=(N, 4)
    Return:
        areas: shape=(N,)
    '''
    w = bboxes[:, 2] - bboxes[:, 0] + 0
    h = bboxes[:, 3] - bboxes[:, 1] + 0
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


if __name__ == "__main__":
    bboxes1 = np.array([[0, 0, 60, 60],
                       [3, 3, 45, 50]], dtype=np.float32)
    bboxes2 = np.array([[10, 10, 25, 25],
                        [22, 12, 100, 100],
                        [32, 12, 50, 80]], dtype=np.float32)
    # bboxes1 = torch.from_numpy(bboxes1)
    # bboxes2 = torch.from_numpy(bboxes2)
    print(box_iou(bboxes1, bboxes2))