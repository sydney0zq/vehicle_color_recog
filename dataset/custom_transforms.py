import numpy as np
import torch
import random
import cv2
cv2.setNumThreads(0)

class IdToLabel(object):
    COLOR_LIST = ["black", "blue", "cyan", "gray", "green", "red", "white", "yellow"]
    def __call__(self, cid):
        return self.COLOR_LIST[cid]


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.p = prob

    def __call__(self, sample):
        if random.random() < self.p:
            sample = cv2.flip(sample, flipCode=1)

        return sample


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):

        h, w = sample.shape[:2]
        new_h, new_w = self.output_size
        new_h = h if new_h >= h else new_h
        new_w = w if new_w >= w else new_w

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        sample = sample[top: top + new_h, left:left + new_w]
        return sample


class RandomScale(object):
    """Randomly resize the image and the ground truth to specified scales.
    Args:
        scales (list): the list of scales
    """

    def __init__(self, min_scale=1., max_scale=1.3, short_edge=None):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.short_edge = short_edge

    def __call__(self, sample):
        sc = np.random.uniform(self.min_scale, self.max_scale)

        if not (self.short_edge is None):
            h, w = sample.shape[:2]
            if h > w:
                sc *= float(self.short_edge) / w
            else:
                sc *= float(self.short_edge) / h

        flagval = cv2.INTER_LINEAR
        sample = cv2.resize(sample, None, fx=sc, fy=sc, interpolation=flagval)
        return sample

class Resize(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def __call__(self, sample):
        h, w = sample.shape[:2]
        if self.output_size == (h, w):
            return sample
        else:
            new_h, new_w = self.output_size

        flagval = cv2.INTER_LINEAR
        sample = cv2.resize(sample, dsize=(new_w, new_h),
                          interpolation=flagval)
        return sample


class ToTensor(object):
    def __call__(self, sample):
        sample = np.asarray(sample, dtype=np.float32)
        sample = sample / 127.5 - 1
        sample = np.transpose(sample, (2, 0, 1))
        sample = torch.from_numpy(sample)
        return sample



# slider = Slider(24)
# ct = np.zeros((30, 100, 100))
# slider(ct)






















