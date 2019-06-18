import nms.py_cpy_nms as py_nms
import nms.cy_cpu_nms as cy_nms

import time


if __name__ == '__main__':
    bboxes = np.array([[0, 0, 30, 30, 0.9],
                       [5, 6, 28, 31, 0.7],
                       [23, 29, 80, 90, 0.6]])

    tic = time.time()
    keep = py_nms.py_cpu_nms(bboxes, 0.2)
    print('pure python => ', keep)
    print('purl python time => ', (time.time() - tic)*1000)

    tic = time.time()
    keep = cy_nms.cy_cpu_nms(bboxes, 0.2)
    print('cython => ', keep)
    print('cython time => ', (time.time() - tic)*1000)