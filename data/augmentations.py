import cv2
import numpy as np


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size,
                                 self.size))
        return image, boxes, labels


class ToTensor(object):
    '''
    convert cv2 image(h, w, c) to (c, h, w), dtype=np.float32
    '''
    def __call__(self, img, boxes=None, labels=None):
        img = np.ascontiguousarray(img.transpose((2, 0, 1))).astype(np.float32)

        return img, boxes, labels


class Norm(object):
    '''
    - mean
    '''
    def __init__(self, mean=(123, 117, 104), std=None):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, img, boxes=None, labels=None):
        img -= self.mean[:, None, None]

        return img, boxes, labels
