from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import albumentations as A
from timm.data import create_transform
from PIL import ImageFilter
import matplotlib.pyplot as plt
import logging
import random
import numpy as np
from PIL import Image
import torchvision.transforms as T
import cv2
import torch

def get_new_shape(images, size):
    w, h = tuple(size)
    shape = list(images.shape)
    shape[1] = h
    shape[2] = w
    shape = tuple(shape)
    return shape

def center_crop(images, size):
    n, _h, _w = images.shape[:3]
    w, h = tuple(size)
    assert (_h>=h and _w>=w)

    y = int(round(0.5 * (_h - h)))
    x = int(round(0.5 * (_w - w)))

    images_new = images[:, y:y+h, x:x+w]

    return images_new

def resize(images, size):
    n, _h, _w = images.shape[:3]
    w, h = tuple(size)
    shape_new = get_new_shape(images, size)

    images_new = np.ndarray(shape_new, dtype=images.dtype)

    for i in range(n):
        images_new[i] = cv2.resize(images[i].astype('float32'), (w,h))

    return images_new

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class DPPRE(object):

    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s2
        self.ts = A.Compose([
            A.PadIfNeeded(int(1.15 * s1), int(1.15 * s2), border_mode=cv2.BORDER_CONSTANT, value=255),
            A.CenterCrop(int(1.15 * s1), int(1.15 * s2)),
            A.Resize(int(1.07 * s1), int(1.07 * s2)),
            A.CenterCrop(s1, s2)
        ], keypoint_params=A.KeypointParams('xya', remove_invisible=True, angle_in_degrees=False, label_fields=["descriptors"]))

    def __call__(self, img, kp, des):
        img = np.array(img)
        img = (255.0 * (img - np.min(img)) / (np.max(img) - np.min(img))).astype(np.uint8)
        transformed = self.ts(image=img, keypoints=kp, descriptors=des)
        img, kp, des = np.array(Image.fromarray(transformed['image']).convert('L')), transformed['keypoints'], transformed['descriptors']

        # standardize
        mean = 127.5
        std = 128.0
        img = img.astype(np.float32)
        img = (img - mean) / std

        img = torch.tensor(np.expand_dims(img, axis=0).repeat(3, axis=0))
        kp, des = np.array(kp), np.array(des)

        return img, kp, des

    def __repr__(self):
        return self.__class__.__name__ + '()'

class DPPREGlobal(object):

    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s2
        self.ts = A.Compose([
            A.PadIfNeeded(int(1.15 * s1), int(1.15 * s2), border_mode=cv2.BORDER_CONSTANT, value=255),
            A.CenterCrop(int(1.15 * s1), int(1.15 * s2)),
            A.Resize(int(1.07 * s1), int(1.07 * s2)),
            A.CenterCrop(s1, s2)
        ])

    def __call__(self, img):
        img = np.array(img)
        transformed = self.ts(image=img)
        img = np.array(Image.fromarray(transformed['image']).convert('L'))

        # standardize
        mean = 127.5
        std = 128.0
        img = img.astype(np.float32)
        img = (img - mean) / std

        img = torch.tensor(np.expand_dims(img, axis=0).repeat(3, axis=0))

        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


def get_resolution(original_resolution):
    """Takes (H,W) and returns (precrop, crop)."""
    area = original_resolution[0] * original_resolution[1]
    return (160, 128) if area < 96*96 else (512, 480)


def build_transforms(cfg, is_train=True):
    if cfg.AUG.TIMM_AUG.USE_TRANSFORM and is_train:
        logging.info('=> use timm transform for training')
        timm_cfg = cfg.AUG.TIMM_AUG
        transforms = create_transform(
            input_size=cfg.TRAIN.IMAGE_SIZE[0],
            is_training=True,
            use_prefetcher=False,
            no_aug=False,
            re_prob=timm_cfg.RE_PROB,
            re_mode=timm_cfg.RE_MODE,
            re_count=timm_cfg.RE_COUNT,
            scale=cfg.AUG.SCALE,
            ratio=cfg.AUG.RATIO,
            hflip=timm_cfg.HFLIP,
            vflip=timm_cfg.VFLIP,
            color_jitter=timm_cfg.COLOR_JITTER,
            auto_augment=timm_cfg.AUTO_AUGMENT,
            interpolation=timm_cfg.INTERPOLATION,
            mean=cfg.INPUT.MEAN,
            std=cfg.INPUT.STD,
        )

        return transforms

    # assert isinstance(cfg.DATASET.OUTPUT_SIZE, (list, tuple)), 'DATASET.OUTPUT_SIZE should be list or tuple'
    normalize = T.Normalize(mean=cfg.INPUT.MEAN, std=cfg.INPUT.STD)

    transforms = None
    if is_train:
        if cfg.FINETUNE.FINETUNE and not cfg.FINETUNE.USE_TRAIN_AUG:
            # precrop, crop = get_resolution(cfg.TRAIN.IMAGE_SIZE)
            crop = cfg.TRAIN.IMAGE_SIZE[0]
            precrop = crop + 32
            transforms = T.Compose([
                T.Resize(
                    (precrop, precrop),
                    interpolation=cfg.AUG.INTERPOLATION
                ),
                T.RandomCrop((crop, crop)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize,
            ])
        else:
            aug = cfg.AUG
            scale = aug.SCALE
            ratio = aug.RATIO
            ts = [
                T.RandomResizedCrop(
                    cfg.TRAIN.IMAGE_SIZE[0], scale=scale, ratio=ratio,
                    interpolation=cfg.AUG.INTERPOLATION
                ),
                T.RandomHorizontalFlip(),
            ]

            cj = aug.COLOR_JITTER
            if cj[-1] > 0.0:
                ts.append(T.RandomApply([T.ColorJitter(*cj[:-1])], p=cj[-1]))

            gs = aug.GRAY_SCALE
            if gs > 0.0:
                ts.append(T.RandomGrayscale(gs))

            gb = aug.GAUSSIAN_BLUR
            if gb > 0.0:
                ts.append(T.RandomApply([GaussianBlur([.1, 2.])], p=gb))

            ts.append(T.ToTensor())
            ts.append(normalize)

            transforms = T.Compose(ts)
    else:
        if cfg.TEST.CENTER_CROP:
            transforms = T.Compose([
                T.CenterCrop(int(cfg.TEST.IMAGE_SIZE[0] * 1.5)),
                T.Resize(
                    int(cfg.TEST.IMAGE_SIZE[0]),
                    interpolation=cfg.TEST.INTERPOLATION
                ),
                T.ToTensor(),
                normalize,
            ])
        elif cfg.TEST.DPPRE == 'global':
            transforms = DPPREGlobal(cfg.TEST.IMAGE_SIZE[1], cfg.TEST.IMAGE_SIZE[0])
        elif cfg.TEST.DPPRE == 'local':
            transforms = DPPRE(cfg.TEST.IMAGE_SIZE[1], cfg.TEST.IMAGE_SIZE[0])
        else:
            transforms = T.Compose([
                T.Resize(
                    (cfg.TEST.IMAGE_SIZE[1], cfg.TEST.IMAGE_SIZE[0]),
                    interpolation=cfg.TEST.INTERPOLATION
                ),
                T.ToTensor(),
                normalize,
            ])

    return transforms
