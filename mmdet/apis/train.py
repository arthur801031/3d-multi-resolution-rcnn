from __future__ import division

from collections import OrderedDict

import torch
from mmcv.runner import Runner, DistSamplerSeedHook
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.core import (DistOptimizerHook, DistEvalmAPHook,
                        CocoDistEvalRecallHook, CocoDistEvalmAPHook,
                        CocoDistEvalmAPHookRGB, CocoDistEvalmAPHookRGB2, CocoDistEvalmAPHook3D, CocoDistEvalmAPHook3DMult,
                        CocoDistEvalmAPHook3DParcel)
from mmdet.datasets import build_dataloader
from mmdet.models import RPN
from .env import get_root_logger


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()

    return loss, log_vars


def batch_processor(model, data, train_mode):
    losses = model(**data)
    loss, log_vars = parse_losses(losses)

    if 'img' in data:
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))
    elif 'imgs' in data:
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['imgs'].data))

    return outputs


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   logger=None,
                   train_dataset2=None):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start training
    if distributed:
        _dist_train(model, dataset, cfg, validate=validate, train_dataset2=train_dataset2)
    else:
        _non_dist_train(model, dataset, cfg, validate=validate)


def _dist_train(model, dataset, cfg, validate=False, train_dataset2=None):
    # prepare data loaders
    if train_dataset2 is not None:
        data_loaders = [
            build_dataloader(
                train_dataset2,
                cfg.data.imgs_per_gpu,
                cfg.data.workers_per_gpu,
                dist=True),
            build_dataloader(
                dataset,
                cfg.data.imgs_per_gpu,
                cfg.data.workers_per_gpu,
                dist=True)
        ]
    else:
        data_loaders = [
            build_dataloader(
                dataset,
                cfg.data.imgs_per_gpu,
                cfg.data.workers_per_gpu,
                dist=True)
        ]
    # put model on gpus
    model = MMDistributedDataParallel(model.cuda())
    # build runner
    runner = Runner(model, batch_processor, cfg.optimizer, cfg.work_dir,
                    cfg.log_level)
    # register hooks
    optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    runner.register_hook(DistSamplerSeedHook())
    # register eval hooks
    if validate:
        if isinstance(model.module, RPN):
            # TODO: implement recall hooks for other datasets
            runner.register_hook(CocoDistEvalRecallHook(cfg.data.val))
        else:
            if cfg.data.val.type == 'CocoDataset':
                runner.register_hook(CocoDistEvalmAPHook(cfg.data.val, cfg.interval))
            elif cfg.data.val.type == 'CocoRGBDataset':
                runner.register_hook(CocoDistEvalmAPHookRGB(cfg.data.val))
            elif cfg.data.val.type == 'CocoDatasetRGB2':
                runner.register_hook(CocoDistEvalmAPHookRGB2(cfg.data.val, cfg.interval))
            elif cfg.data.val.type == 'Coco3DDataset' and hasattr(cfg, 'data2') and hasattr(cfg.data2, 'val'):
                runner.register_hook(CocoDistEvalmAPHook3DMult(cfg.data.val, cfg.data2.val))
            elif cfg.data.val.type == 'Coco3DDataset':
                runner.register_hook(CocoDistEvalmAPHook3D(cfg.data.val, cfg.interval))
            elif cfg.data.val.type == 'Coco3DParcelDataset':
                runner.register_hook(CocoDistEvalmAPHook3DParcel(cfg.data.val, cfg.interval))
            elif cfg.data.val.type == 'Coco3D2ScalesDataset':
                runner.register_hook(CocoDistEvalmAPHook3D(cfg.data.val, cfg.interval, dataset2=cfg.data2_2scales.val))
            elif cfg.data.val.type == 'Coco3D3ScalesDataset':
                runner.register_hook(CocoDistEvalmAPHook3D(cfg.data.val, cfg.interval, dataset3=cfg.data3_3scales.val))
            else:
                runner.register_hook(DistEvalmAPHook(cfg.data.val))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def _non_dist_train(model, dataset, cfg, validate=False):
    # prepare data loaders
    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False)
    ]
    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
    # build runner
    runner = Runner(model, batch_processor, cfg.optimizer, cfg.work_dir,
                    cfg.log_level)
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
