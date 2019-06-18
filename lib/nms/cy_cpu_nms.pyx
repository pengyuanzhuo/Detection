import numpy as np
cimport numpy as np
import cython


def cy_cpu_nms(np.ndarray[np.float32_t, ndim=2] bboxes, np.float threshold):
    '''
    cython cpu nms
    Args:
        bboxes: shape=(n, 5), n is the num of boxes
            vstack of [x1, y1, x2, y2, score]
        threshold: float. iou threshold

    Return:
        keep_indices: indices of bboxes(keeped)
    '''
    cdef np.ndarray[np.float32_t, ndim=1] x1 = bboxes[:, 0]
    cdef np.ndarray[np.float32_t, ndim=1] y1 = bboxes[:, 1]
    cdef np.ndarray[np.float32_t, ndim=1] x2 = bboxes[:, 2]
    cdef np.ndarray[np.float32_t, ndim=1] y2 = bboxes[:, 3]
    cdef np.ndarray[np.float32_t, ndim=1] scores = bboxes[:, 4]

    cdef np.ndarray[np.float32_t, ndim=1] areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    cdef np.ndarray[np.int32_t, ndim=1] des_indices = np.argsort(scores)[::-1]

    keep = []
    cdef int cur_indices
    cdef np.ndarray[np.float32_t, ndim=1] x1_i
    cdef np.ndarray[np.float32_t, ndim=1] y1_i
    cdef np.ndarray[np.float32_t, ndim=1] x2_i
    cdef np.ndarray[np.float32_t, ndim=1] y2_i

    cdef np.ndarray[np.float32_t, ndim=1] w
    cdef np.ndarray[np.float32_t, ndim=1] h
    cdef np.ndarray[np.float32_t, ndim=1] inter, union, iou, keep_indices

    while des_indices.size > 0:
        cur_indices = des_indices[0]
        keep.append(cur_indices)

        x1_i = np.maximum(x1[cur_indices], x1[des_indices[1:]])
        y1_i = np.maximum(y1[cur_indices], y1[des_indices[1:]])
        x2_i = np.minimum(x2[cur_indices], x2[des_indices[1:]])
        y2_i = np.minimum(y2[cur_indices], y2[des_indices[1:]])

        w = np.maximum(0.0, x2_i - x1_i + 1)
        h = np.maximum(0.0, y2_i - y1_i + 1)

        inter = w * h
        union = areas[cur_indices] + areas[des_indices[1:]] - inter

        iou = inter / union

        keep_indices = np.where(iou <= threshold)[0]
        des_indices = des_indices[keep_indices + 1]

    return keep
