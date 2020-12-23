import argparse

import torch
import mmcv
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel

from mmdet import datasets
from mmdet.core import results2json, coco_eval, results2json3D, results2json3DMulti, results2json3DParcel
from mmdet.datasets import build_dataloader
from mmdet.models import build_detector, detectors
from pycocotools_local.coco import *


def single_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        # if i > 5:
        #     return results
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result, dataset.img_norm_cfg,
                                     dataset=dataset.CLASSES)
        if 'img' in data:
            batch_size = data['img'][0].size(0)
        elif 'imgs' in data:
            batch_size = data['imgs'].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results

def double_test(model, data_loader, data_loader2, test_cfg2, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    dataset2 = data_loader2.dataset
    prog_bar = mmcv.ProgressBar(len(dataset) + len(dataset2))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result, dataset.img_norm_cfg,
                                     dataset=dataset.CLASSES)
        if 'img' in data:
            batch_size = data['img'][0].size(0)
        elif 'imgs' in data:
            batch_size = data['imgs'].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    
    for i, data in enumerate(data_loader2):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, test_cfg2=test_cfg2, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result, dataset.img_norm_cfg,
                                     dataset=dataset.CLASSES)
        if 'img' in data:
            batch_size = data['img'][0].size(0)
        elif 'imgs' in data:
            batch_size = data['imgs'].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results

def _data_func(data, device_id):
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(return_loss=False, rescale=True, **data)


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--gpus', default=1, type=int, help='GPU number used for testing')
    parser.add_argument(
        '--proc_per_gpu',
        default=1,
        type=int,
        help='Number of processes per GPU')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))
    if args.gpus == 1:
        model = build_detector(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        load_checkpoint(model, args.checkpoint)
        model = MMDataParallel(model, device_ids=[0])

        if hasattr(cfg, 'data2'):
            dataset2 = obj_from_dict(cfg.data2.test, datasets, dict(test_mode=True))
            data_loader2 = build_dataloader(
                dataset2,
                imgs_per_gpu=1,
                workers_per_gpu=cfg.data2.workers_per_gpu,
                num_gpus=1,
                dist=False,
                shuffle=False)
            data_loader = build_dataloader(
                dataset,
                imgs_per_gpu=1,
                workers_per_gpu=cfg.data.workers_per_gpu,
                num_gpus=1,
                dist=False,
                shuffle=False)
            outputs = double_test(model, data_loader, data_loader2, cfg.test_cfg2, args.show)
        else:
            data_loader = build_dataloader(
                dataset,
                imgs_per_gpu=1,
                workers_per_gpu=cfg.data.workers_per_gpu,
                num_gpus=1,
                dist=False,
                shuffle=False)
            outputs = single_test(model, data_loader, args.show)
    else:
        model_args = cfg.model.copy()
        model_args.update(train_cfg=None, test_cfg=cfg.test_cfg)
        model_type = getattr(detectors, model_args.pop('type'))
        outputs = parallel_test(
            model_type,
            model_args,
            args.checkpoint,
            dataset,
            _data_func,
            range(args.gpus),
            workers_per_gpu=args.proc_per_gpu)

    if args.out:
        # print('writing results to {}'.format(args.out))
        # mmcv.dump(outputs, args.out)
        eval_types = args.eval
        if eval_types:
            print('Starting evaluate {}'.format(' and '.join(eval_types)))
            if eval_types == ['proposal_fast']:
                result_file = args.out
                coco_eval(result_file, eval_types, dataset.coco)
            else:
                if not isinstance(outputs[0], dict):
                    result_file = args.out + '.json'
                    # 3D
                    # load full volume and get full volume's image IDs
                    if hasattr(cfg.data.test, 'ann_file_volume'):
                        coco_full_gt = COCO(cfg.data.test.ann_file_volume)
                    else:
                        coco_full_gt = COCO(cfg.data.test.ann_file)

                    if str(type(dataset)) == "<class 'mmdet.datasets.coco_3d.Coco3DDataset'>" or \
                        str(type(dataset)) == "<class 'mmdet.datasets.coco_3d_2scales.Coco3D2ScalesDataset'>" or \
                        str(type(dataset)) == "<class 'mmdet.datasets.coco_3d_3scales.Coco3D3ScalesDataset'>" or \
                        str(type(dataset)) == "<class 'mmdet.datasets.coco_3d_parcel.Coco3DParcelDataset'>":
                        full_filename_to_id = dict()
                        for img_id in coco_full_gt.getImgIds():
                            full_filename_to_id[coco_full_gt.loadImgs([img_id])[0]['file_name']] = img_id

                        if cfg.data.test.with_mask:
                            if hasattr(cfg, 'data2') and hasattr(cfg.data2, 'test'):
                                result = results2json3DMulti(dataset, dataset2, outputs, result_file, full_filename_to_id)
                            else:
                                result = results2json3D(dataset, outputs, result_file, full_filename_to_id)
                            coco_eval(result, eval_types, coco_full_gt, is3D=True, hasMask=True, full_filename_to_id=full_filename_to_id)
                        else:
                            if hasattr(cfg, 'data2') and hasattr(cfg.data2, 'test'):
                                results2json3DMulti(dataset, dataset2, outputs, result_file, full_filename_to_id)
                                coco_eval(result_file, eval_types, coco_full_gt, is3D=True, full_filename_to_id=full_filename_to_id)
                            elif str(type(dataset)) == "<class 'mmdet.datasets.coco_3d_parcel.Coco3DParcelDataset'>":
                                results2json3DParcel(dataset, outputs, result_file, full_filename_to_id)
                                coco_eval(result_file, eval_types, coco_full_gt, is3D=True, full_filename_to_id=full_filename_to_id, isParcellized=True)
                            else:
                                results2json3D(dataset, outputs, result_file, full_filename_to_id)
                                coco_eval(result_file, eval_types, coco_full_gt, is3D=True, full_filename_to_id=full_filename_to_id)
                    else:
                        # default
                        results2json(dataset, outputs, result_file)
                        coco_eval(result_file, eval_types, dataset.coco)
                else:
                    for name in outputs[0]:
                        print('\nEvaluating {}'.format(name))
                        outputs_ = [out[name] for out in outputs]
                        result_file = args.out + '.{}.json'.format(name)
                        results2json(dataset, outputs_, result_file)
                        coco_eval(result_file, eval_types, dataset.coco)


if __name__ == '__main__':
    main()
