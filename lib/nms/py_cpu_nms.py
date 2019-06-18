# coding: utf-8

import numpy as np


def py_cpu_nms(bboxes, threshold):
    '''
    python cpu nms baseline
    Args:
        bboxes: shape=(n, 5), n is the num of boxes
            vstack of [x1, y1, x2, y2, score]
        threshold: float. iou threshold

    Return:
        keep_indices: indices of bboxes(keeped)
    '''
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    scores = bboxes[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1) # shape = (1, n)

    des_indices = np.argsort(scores)[::-1]
    keep = []

    while des_indices.size > 0:
        cur_indices = des_indices[0]
        keep.append(cur_indices)

        x1_i = np.maximum(x1[cur_indices], x1[des_indices[1:]]) # current vs rest
        y1_i = np.maximum(y1[cur_indices], y1[des_indices[1:]])
        x2_i = np.minimum(x2[cur_indices], x2[des_indices[1:]])
        y2_i = np.minimum(y2[cur_indices], y2[des_indices[1:]])

        w = np.maximum(0.0, x2_i - x1_i + 1)
        h = np.maximum(0.0, y2_i - y1_i + 1)

        inter = w * h # n-1
        union = areas[cur_indices] + areas[des_indices[1:]] - inter # n-1

        iou = inter / union

        keep_indices = np.where(iou <= threshold)[0]
        des_indices = des_indices[keep_indices + 1]

    return keep


if __name__ == '__main__':
    import time
    bboxes = np.array([[0, 0, 30, 30, 0.9],
                       [5, 6, 28, 31, 0.7],
                       [23, 29, 80, 90, 0.6]])

    tic = time.time()
    keep = py_cpu_nms(bboxes, 0.2)
    print(keep)
    print('time => ', (time.time() - tic)*1000)
