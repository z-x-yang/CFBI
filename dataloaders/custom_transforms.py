import os
import random
import cv2
import numpy as np
import torch

cv2.setNumThreads(0)

class MultiRestrictSize(object):
    def __init__(self, min_size=None, max_size=800, flip=False, multi_scale=[1.3]):
        self.min_size = min_size
        self.max_size = max_size
        self.multi_scale = multi_scale
        self.flip = flip
        assert ((min_size is None)) or ((max_size is None))

    def __call__(self, sample):
        samples = []
        image = sample['current_img']
        h, w = image.shape[:2]
        for scale in self.multi_scale:
            # Fixed range of scales
            sc = None
            # Align short edge
            if not (self.min_size is None):
                if h > w:
                    short_edge = w
                else:
                    short_edge = h
                if short_edge < self.min_size:
                    sc = float(self.min_size) / short_edge
            else:
                if h > w:
                    long_edge = h
                else:
                    long_edge = w
                if long_edge > self.max_size:
                    sc = float(self.max_size) / long_edge

            if sc is None:
                new_h = h
                new_w = w
            else:
                new_h = sc * h
                new_w = sc * w
            new_h = int(new_h * scale)
            new_w = int(new_w * scale)

            if (new_h - 1) % 16 != 0:
                new_h = int(np.around((new_h - 1) / 16.) * 16 + 1)
            if (new_w - 1) % 16 != 0:
                new_w = int(np.around((new_w - 1) / 16.) * 16 + 1)
                
            if new_h == h and new_w == w:
                samples.append(sample)
            else:
                new_sample = {}
                for elem in sample.keys():
                    if 'meta' in elem:
                        new_sample[elem] = sample[elem]
                        continue
                    tmp = sample[elem]
                    if 'label' in elem:
                        new_sample[elem] = sample[elem]
                        continue
                    else:
                        flagval = cv2.INTER_CUBIC
                        tmp = cv2.resize(tmp, dsize=(new_w, new_h), interpolation=flagval)
                        new_sample[elem] = tmp
                samples.append(new_sample)

            if self.flip:
                now_sample = samples[-1]
                new_sample = {}
                for elem in now_sample.keys():
                    if 'meta' in elem:
                        new_sample[elem] = now_sample[elem].copy()
                        new_sample[elem]['flip'] = True
                        continue
                    tmp = now_sample[elem]
                    tmp = tmp[:, ::-1].copy()
                    new_sample[elem] = tmp
                samples.append(new_sample)

        return samples

class MultiToTensor(object):
    def __call__(self, samples):
        for idx in range(len(samples)):
            sample = samples[idx]
            for elem in sample.keys():
                if 'meta' in elem:
                    continue
                tmp = sample[elem]
                if tmp is None:
                    continue

                if tmp.ndim == 2:
                    tmp = tmp[:, :, np.newaxis]
                else:
                    tmp = tmp / 255.
                    tmp -= (0.485, 0.456, 0.406)
                    tmp /= (0.229, 0.224, 0.225)

                tmp = tmp.transpose((2, 0, 1))
                samples[idx][elem] = torch.from_numpy(tmp)

        return samples

        