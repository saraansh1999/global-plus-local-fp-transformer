from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import sys
import re
import logging
import time
import torch
import os
import numpy as np
import pickle
from scipy import special
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from timm.data import Mixup
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torchvision

from core.evaluate import accuracy
from utils.comm import comm


def train_one_epoch(config, train_loader, model, criterion, optimizer, epoch,
                    output_dir, tb_log_dir, writer_dict, scaler=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    logging.info('=> switch to train mode')
    model.train()
    criterion.train()

    aug = config.AUG

    end = time.time()
    for i, (inputs) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}

        with autocast(enabled=config.AMP.ENABLED):
            outputs = model(inputs['img'])
            loss_dict, loss = criterion(outputs, inputs)

        if i == 0:
            loss_recorder = {k: AverageMeter() for k in loss_dict.keys()}

        # compute gradient and do update step
        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') \
            and optimizer.is_second_order

        scaler.scale(loss).backward(create_graph=is_second_order)

        if config.TRAIN.CLIP_GRAD_NORM > 0.0:
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)

            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.TRAIN.CLIP_GRAD_NORM
            )

        scaler.step(optimizer)
        scaler.update()

        # measure accuracy and record loss
        for k in loss_dict.keys():
            loss_recorder[k].update(loss_dict[k].item(), inputs['img'].size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = '\n => Epoch[{0}][{1}/{2}]: ' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'.format(
                      epoch, i, len(train_loader),
                      batch_time=batch_time,
                      speed=inputs['img'].size(0)/batch_time.val,
                      data_time=data_time)
            for k in loss_recorder.keys():
                msg += '\t' + k + ': ' + str(loss_recorder[k].avg)
            logging.info(msg)

        torch.cuda.synchronize()

    if writer_dict and comm.is_main_process():
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        for k in loss_recorder.keys():
            writer.add_scalar('train_'+k, loss_recorder[k].avg, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1
        img_grid = torchvision.utils.make_grid((255 * (inputs['img'][:6])).type(torch.uint8))
        writer.add_image('train_images', img_grid)


@torch.no_grad()
def test(config, val_loader, model, criterion, output_dir, tb_log_dir,
         writer_dict=None, distributed=False, real_labels=None,
         valid_labels=None):
    batch_time = AverageMeter()

    logging.info('=> switch to eval mode')
    model.eval()
    criterion.eval()

    end = time.time()
    for i, (inputs) in enumerate(val_loader):

        inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}

        outputs = model(inputs['img'])
        loss_dict, loss = criterion(outputs, inputs)

        # measure accuracy and record loss
        if i == 0:
            loss_recorder = {k: AverageMeter() for k in loss_dict.keys()}
        for k in loss_dict.keys():
            loss_recorder[k].update(loss_dict[k].item(), inputs['img'].size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logging.info('=> synchronize...')
    comm.synchronize()
    if distributed:
        for k in loss_recorder.keys():
            loss_recorder[k] =  _meter_reduce(loss_recorder[k])
    else:
        for k in loss_recorder.keys():
            loss_recorder[k] = loss_recorder[k].avg

    if comm.is_main_process():
        msg = '=> TEST:\t'
        for k in loss_recorder.keys():
            msg += '\t' + k + ': ' + str(loss_recorder[k])
        logging.info(msg)

    if writer_dict and comm.is_main_process():
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        for k in loss_recorder.keys():
            writer.add_scalar('valid_'+k, loss_recorder[k], global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1
        img_grid = torchvision.utils.make_grid((255 * (inputs['img'][:6])).type(torch.uint8))
        writer.add_image('val_images', img_grid)

    logging.info('=> switch to train mode')

    return loss_recorder[config.PERF_KEY]

@torch.no_grad()
def inference(config, val_loader, model, criterion, output_dir, tb_log_dir,
         writer_dict=None, distributed=False):
    batch_time = AverageMeter()

    logging.info('=> switch to eval mode')
    model.eval()
    criterion.eval()

    identifiers = []
    results = {}

    end = time.time()

    for i, (inputs) in enumerate(val_loader):

        # compute output
        for k, v in inputs.items():
            if not isinstance(v, list):
                inputs[k] = v.cuda(non_blocking=True)

        outputs = model(inputs['img'])

        loss_dict, loss = criterion(outputs, inputs)

        if config.MODEL.RET_DEC_INTERMEDIATE:
            outputs['kd_tokens'] = outputs['kd_tokens'][-1]
            outputs['posori_tokens'] = outputs['posori_tokens'][-1]

        # measure accuracy and record loss
        if i == 0:
            loss_recorder = {k: AverageMeter() for k in loss_dict.keys()}
        for k in loss_dict.keys():
            loss_recorder[k].update(loss_dict[k].item(), inputs['img'].size(0))

        if len(identifiers) == 0:
            identifiers = inputs['key']
            if 'kd_tokens' in outputs:
                results['embs'] = outputs['kd_tokens'].cpu().numpy()
                results['posori'] = outputs['posori_tokens'].cpu().numpy()
            if 'kd_global' in outputs:
                results['global'] = outputs['kd_global'].cpu().numpy()
        else:
            identifiers = np.concatenate((identifiers, inputs['key']))
            if 'kd_tokens' in outputs:
                results['embs'] = np.concatenate((results['embs'], outputs['kd_tokens'].cpu().numpy()))
                results['posori'] = np.concatenate((results['posori'], outputs['posori_tokens'].cpu().numpy()))
            if 'kd_global' in outputs:
                results['global'] = np.concatenate((results['global'], outputs['kd_global'].cpu().numpy()))

        # measure accuracy and record loss
        if i == 0:
            loss_recorder = {k: AverageMeter() for k in loss_dict.keys()}
        for k in loss_dict.keys():
            loss_recorder[k].update(loss_dict[k].item(), inputs['img'].size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logging.info('=> synchronize...')
    comm.synchronize()
    if distributed:
        for k in loss_recorder.keys():
            loss_recorder[k] =  _meter_reduce(loss_recorder[k])
    else:
        for k in loss_recorder.keys():
            loss_recorder[k] = loss_recorder[k].avg

    if 'posori'in results:
        for i in range(identifiers.shape[0]):
            subject = '_'.join(identifiers[i].split('_')[:-1])
            in_file = os.path.join(config.DATASET.VAL_TOKEN_EMBS, subject, identifiers[i]+'.npy')
            in_file = np.load(in_file, allow_pickle=True, encoding='bytes').item()

            if b'patches' in in_file:
                del in_file[b'patches']

            in_file[b'mnt'] = results['posori'][i]
            in_file[b'mnt'][:, 0] = (in_file[b'mnt'][:, 0] * config.TEST.IMAGE_SIZE[0]).astype(int)
            in_file[b'mnt'][:, 1] = (in_file[b'mnt'][:, 1] * config.TEST.IMAGE_SIZE[1]).astype(int)
            in_file[b'mnt'][:, 2] = in_file[b'mnt'][:, 2] * np.pi * 2
            in_file[b'w'] = config.TEST.IMAGE_SIZE[0]
            in_file[b'h'] = config.TEST.IMAGE_SIZE[1]
            in_file[b'mnt'] = np.concatenate( (in_file[b'mnt'], np.zeros((in_file[b'mnt'].shape[0], 1)) ), axis=1)

            in_file[b'des'] = np.expand_dims(results['embs'][i], axis=0)

            save_path = os.path.join(config.OUTPUT_DIR, subject, identifiers[i]+'.pkl')
            pardir = '/'.join(save_path.split('/')[:-1])
            if not os.path.exists(pardir):
                os.makedirs(pardir)
            with open(save_path, 'wb') as handle:
                pickle.dump(in_file, handle, protocol=2)

    if 'global' in results:
        global_embs = {}
        for i in range(identifiers.shape[0]):
            global_embs[identifiers[i]] = results['global'][i]
        np.save(os.path.join(config.OUTPUT_DIR, 'global_embs.npy'), global_embs)

    if comm.is_main_process():
        msg = '=> TEST:\t'
        for k in loss_recorder.keys():
            msg += '\t' + k + ': ' + str(loss_recorder[k])
        logging.info(msg)

    logging.info('=> switch to train mode')
    model.train()

@torch.no_grad()
def only_forward(config, val_loader, model):
    batch_time = AverageMeter()

    logging.info('=> switch to eval mode')
    model.eval()

    results = {}

    end = time.time()

    for i, (inputs) in enumerate(val_loader):
        
        print("Starting batch", i, "...")
        outputs = model(inputs['img'])
        
        if config.MODEL.RET_DEC_INTERMEDIATE:
            if 'kd_tokens' in outputs:
                outputs['kd_tokens'] = outputs['kd_tokens'][-1]
            if 'posori_tokens' in outputs:
                outputs['posori_tokens'] = outputs['posori_tokens'][-1]
            
        if len(results) == 0:
            if 'kd_tokens' in outputs:
                results['embs'] = outputs['kd_tokens'].cpu().numpy()
                results['posori'] = outputs['posori_tokens'].cpu().numpy()
            if 'kd_global' in outputs:
                results['global'] = outputs['kd_global'].cpu().numpy()
                if len(results['global'].shape) == 1:
                    results['global'] = results['global'].reshape(1, -1)
        else:
            if 'kd_tokens' in outputs:
                results['embs'] = np.concatenate((results['embs'], outputs['kd_tokens'].cpu().numpy()))
                results['posori'] = np.concatenate((results['posori'], outputs['posori_tokens'].cpu().numpy()))
            if 'kd_global' in outputs:
                if len(outputs['kd_global'].shape) == 1:
                    outputs['kd_global'] = outputs['kd_global'].reshape(1, -1)
                results['global'] = np.concatenate((results['global'], outputs['kd_global'].cpu().numpy()), axis=0)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return results
    

def _meter_reduce(meter):
    rank = comm.local_rank
    meter_sum = torch.FloatTensor([meter.sum]).cuda(rank)
    meter_count = torch.FloatTensor([meter.count]).cuda(rank)
    torch.distributed.reduce(meter_sum, 0)
    torch.distributed.reduce(meter_count, 0)
    meter_avg = meter_sum / meter_count

    return meter_avg.item()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



