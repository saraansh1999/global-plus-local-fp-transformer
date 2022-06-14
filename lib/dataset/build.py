from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch._six import container_abcs, string_classes, int_classes
from torch.utils.data.dataloader import default_collate
import logging
import random
import os
import sys
import re

from timm.data import create_loader
import torch
import torch.utils.data
import torchvision.datasets as datasets

from .transformas import build_transforms
from .samplers import RASampler

from PIL import Image
import numpy as np

def is_spoof_file(path):
    return ('Fake' in path or 'Spoof' in path or 'fake' in path or 'spoof' in path)

def build_dataset(cfg, is_train):
    dataset = None
    if is_train:
        if 'imagenet' in cfg.DATASET.TRAIN_DATASET:
            dataset = _build_imagenet_dataset(cfg, is_train)
        elif 'cvtdetr' in cfg.DATASET.TRAIN_DATASET:
           dataset = CustomDatasetCVTDETR(cfg, is_train)
        else:
            raise ValueError('Unkown dataset: {}'.format(cfg.DATASET.DATASET))
    else:
        if 'imagenet' in cfg.DATASET.VAL_DATASET:
            dataset = _build_imagenet_dataset(cfg, is_train)
        elif 'cvtdetr' in cfg.DATASET.VAL_DATASET:
           dataset = CustomDatasetCVTDETR(cfg, is_train)
        elif 'inference' in cfg.DATASET.VAL_DATASET:
           dataset = CustomDatasetInference(cfg, is_train)
        else:
            raise ValueError('Unkown dataset: {}'.format(cfg.DATASET.DATASET))
    return dataset

class CustomDatasetCVTDETR(torch.utils.data.Dataset):
    def __init__(self, cfg, is_train):
        self.transforms = build_transforms(cfg, is_train=False)         # we dont want augs

        if is_train:
            data_dir = cfg.DATASET.TRAIN_IMGS
            global_emb_dir = cfg.DATASET.TRAIN_GLOBAL_EMBS
            token_emb_dir = cfg.DATASET.TRAIN_TOKEN_EMBS
            self.img_sz = cfg.TRAIN.IMAGE_SIZE
        else:
            data_dir = cfg.DATASET.VAL_IMGS
            global_emb_dir = cfg.DATASET.VAL_GLOBAL_EMBS
            token_emb_dir = cfg.DATASET.VAL_TOKEN_EMBS
            self.img_sz = cfg.TEST.IMAGE_SIZE

        self.max_mnt = cfg.DATASET.MAX_MNT
        self.global_req = cfg.MODEL.RET_GLOBAL
        self.local_req = cfg.MODEL.RET_LOCAL

        self.num = 0
        self.imgs, self.global_embs, self.token_embs = [], [], []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                fpath = os.path.join(root, file)
                if is_spoof_file(fpath):
                    continue
                gepath = os.path.join(global_emb_dir, fpath[len(data_dir.rstrip('/'))+1:]).split('.')[0] + '.npy'
                tepath = os.path.join(token_emb_dir, fpath[len(data_dir.rstrip('/'))+1:]).split('.')[0] + '.npy'
                if (os.path.exists(gepath) or not self.global_req) and (os.path.exists(tepath) or not self.local_req):
                    self.imgs.append(fpath)
                    self.global_embs.append(gepath)
                    self.token_embs.append(tepath)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, ind):
        inputs = {}

        img = Image.open(self.imgs[ind]).convert('RGB')

        if self.global_req:
            global_emb = np.load(self.global_embs[ind], allow_pickle=True)
            inputs['global_emb'] = global_emb.astype(np.float32)

        if self.local_req:
            token_emb = np.load(self.token_embs[ind], allow_pickle=True, encoding='bytes').item()
            local_emb = np.squeeze(token_emb[b'des'])
            mnt = token_emb[b'mnt']

            # sorting to get the best minutiae first
            inds = np.argsort(-mnt[:, -1])
            mnt = mnt[inds, :-1]
            local_emb = local_emb[inds]

            img, mnt, local_emb = self.transforms(img, mnt, local_emb)

            # changing the range of values
            mnt[:, 0] = mnt[:, 0] / self.img_sz[0]
            mnt[:, 1] = mnt[:, 1] / self.img_sz[1]
            mnt[:, 2] = mnt[:, 2] / (2 * np.pi)

            # restricting max num of minutiae
            num_mnt = mnt.shape[0]
            if num_mnt > self.max_mnt:
                local_emb = local_emb[:self.max_mnt]
                mnt = mnt[:self.max_mnt]
            elif num_mnt < self.max_mnt:
                num_pad = self.max_mnt - num_mnt
                pad = np.zeros((num_pad, local_emb.shape[1]), dtype=local_emb.dtype)
                local_emb = np.concatenate((local_emb, pad))
                pad = np.zeros((num_pad, mnt.shape[1]), dtype=mnt.dtype)
                mnt = np.concatenate((mnt, pad))

            inputs['local_emb'] = local_emb.astype(np.float32)
            inputs['mnt'] = mnt.astype(np.float32)
            inputs['num_mnt'] = min(self.max_mnt, num_mnt)
        else:
            img = self.transforms(img)

        inputs['img'] = img

        return inputs


class CustomDatasetInference(torch.utils.data.Dataset):
    def __init__(self, cfg, is_train):
        self.transforms = build_transforms(cfg, is_train=False)         # we dont want augs

        data_dir = cfg.DATASET.VAL_IMGS
        global_emb_dir = cfg.DATASET.VAL_GLOBAL_EMBS
        token_emb_dir = cfg.DATASET.VAL_TOKEN_EMBS
        self.img_sz = cfg.TEST.IMAGE_SIZE

        self.max_mnt = cfg.DATASET.MAX_MNT
        self.global_req = cfg.MODEL.RET_GLOBAL
        self.local_req = cfg.MODEL.RET_LOCAL

        self.imgs, self.global_embs, self.token_embs, self.keys = [], [], [], []

        for root, dirs, files in os.walk(data_dir):
            for file in files:
                fpath = os.path.join(root, file)
                gepath = os.path.join(global_emb_dir, fpath[len(data_dir.rstrip('/'))+1:]).split('.')[0] + '.npy'
                tepath = os.path.join(token_emb_dir, fpath[len(data_dir.rstrip('/'))+1:]).split('.')[0] + '.npy'
                if (os.path.exists(gepath) or not self.global_req) and (os.path.exists(tepath) or not self.local_req):
                    self.imgs.append(fpath)
                    self.global_embs.append(gepath)
                    self.token_embs.append(tepath)
                    self.keys.append(fpath.split('/')[-1].split('.')[0])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, ind):
        inputs = {}

        img = Image.open(self.imgs[ind]).convert('RGB')

        key = self.keys[ind]
        inputs['key'] = key

        if self.global_req:
            global_emb = np.load(self.global_embs[ind], allow_pickle=True)
            inputs['global_emb'] = global_emb.astype(np.float32)

        if self.local_req:
            token_emb = np.load(self.token_embs[ind], allow_pickle=True, encoding='bytes').item()
            local_emb = np.squeeze(token_emb[b'des'])
            mnt = token_emb[b'mnt']

            # sorting to get the best minutiae first
            inds = np.argsort(-mnt[:, -1])
            mnt = mnt[inds, :-1]
            local_emb = local_emb[inds]

            img, mnt, local_emb = self.transforms(img, mnt, local_emb)

            # changing the range of values
            mnt[:, 0] = mnt[:, 0] / self.img_sz[0]
            mnt[:, 1] = mnt[:, 1] / self.img_sz[1]
            mnt[:, 2] = mnt[:, 2] / (2 * np.pi)

            # restricting max num of minutiae
            num_mnt = mnt.shape[0]
            if num_mnt > self.max_mnt:
                local_emb = local_emb[:self.max_mnt]
                mnt = mnt[:self.max_mnt]
            elif num_mnt < self.max_mnt:
                num_pad = self.max_mnt - num_mnt
                pad = np.zeros((num_pad, local_emb.shape[1]), dtype=local_emb.dtype)
                local_emb = np.concatenate((local_emb, pad))
                pad = np.zeros((num_pad, mnt.shape[1]), dtype=mnt.dtype)
                mnt = np.concatenate((mnt, pad))

            inputs['local_emb'] = local_emb.astype(np.float32)
            inputs['mnt'] = mnt.astype(np.float32)
            inputs['num_mnt'] = min(self.max_mnt, num_mnt)
        else:
            img = self.transforms(img)

        inputs['img'] = img

        return inputs

def _build_image_folder_dataset(cfg, is_train):
    transforms = build_transforms(cfg, is_train)

    dataset_name = cfg.DATASET.TRAIN_SET if is_train else cfg.DATASET.TEST_SET
    dataset = datasets.ImageFolder(
        os.path.join(cfg.DATASET.ROOT, dataset_name), transforms
    )
    logging.info(
        '=> load samples: {}, is_train: {}'
        .format(len(dataset), is_train)
    )

    return dataset


def _build_imagenet_dataset(cfg, is_train):
    transforms = build_transforms(cfg, is_train)

    dataset_name = cfg.DATASET.TRAIN_SET if is_train else cfg.DATASET.TEST_SET
    dataset = datasets.ImageFolder(
        os.path.join(cfg.DATASET.ROOT, dataset_name), transforms
    )

    return dataset


def custom_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    batch = list(filter(lambda x : x is not None, batch))
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return custom_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: custom_collate([d[key] for d in batch if ('num_kp' not in d) or (d['num_kp'] > 0)]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(custom_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [custom_collate(samples) for samples in transposed]

    raise TypeError(elem_type) #default_collate_err_msg_format.format(elem_type))

def build_dataloader(cfg, is_train=True, distributed=False):
    if is_train:
        batch_size_per_gpu = cfg.TRAIN.BATCH_SIZE_PER_GPU
        shuffle = True
    else:
        batch_size_per_gpu = cfg.TEST.BATCH_SIZE_PER_GPU
        shuffle = False

    dataset = build_dataset(cfg, is_train)
    if is_train:
        print("# data points in train dataset: ", len(dataset))
    else:
        print("# data points in test/val dataset: ", len(dataset))
    if distributed:
        if is_train and cfg.DATASET.SAMPLER == 'repeated_aug':
            logging.info('=> use repeated aug sampler')
            sampler = RASampler(dataset, shuffle=shuffle)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=shuffle
            )
        shuffle = False
    else:
        sampler = None

    if cfg.AUG.TIMM_AUG.USE_LOADER and is_train:
        logging.info('=> use timm loader for training')
        timm_cfg = cfg.AUG.TIMM_AUG
        data_loader = create_loader(
            dataset,
            input_size=cfg.TRAIN.IMAGE_SIZE[0],
            batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
            is_training=True,
            use_prefetcher=True,
            no_aug=False,
            re_prob=timm_cfg.RE_PROB,
            re_mode=timm_cfg.RE_MODE,
            re_count=timm_cfg.RE_COUNT,
            re_split=timm_cfg.RE_SPLIT,
            scale=cfg.AUG.SCALE,
            ratio=cfg.AUG.RATIO,
            hflip=timm_cfg.HFLIP,
            vflip=timm_cfg.VFLIP,
            color_jitter=timm_cfg.COLOR_JITTER,
            auto_augment=timm_cfg.AUTO_AUGMENT,
            num_aug_splits=0,
            interpolation=timm_cfg.INTERPOLATION,
            mean=cfg.INPUT.MEAN,
            std=cfg.INPUT.STD,
            num_workers=cfg.WORKERS,
            distributed=distributed,
            collate_fn=None,
            pin_memory=cfg.PIN_MEMORY,
            use_multi_epochs_loader=True
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size_per_gpu,
            shuffle=shuffle,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            sampler=sampler,
            drop_last=is_train,
            collate_fn=custom_collate
        )

    return data_loader
