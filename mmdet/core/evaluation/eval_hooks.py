import os
import os.path as osp

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import Hook, obj_from_dict
from mmcv.parallel import scatter, collate
from pycocotools.cocoeval import COCOeval
from pycocotools_local.coco import *
from torch.utils.data import Dataset

from .coco_utils import results2json, fast_eval_recall, results2jsonRGB, results2json3D, results2json3DMulti
from .mean_ap import eval_map
from mmdet import datasets


class DistEvalHook(Hook):

    def __init__(self, dataset, interval=1):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = obj_from_dict(dataset, datasets,
                                         {'test_mode': True})
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.interval = interval

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()
        results = [None for _ in range(len(self.dataset))]
        if runner.rank == 0:
            prog_bar = mmcv.ProgressBar(len(self.dataset))
        for idx in range(runner.rank, len(self.dataset), runner.world_size):
            data = self.dataset[idx]
            data_gpu = scatter(
                collate([data], samples_per_gpu=1),
                [torch.cuda.current_device()])[0]

            # compute output
            with torch.no_grad():
                result = runner.model(
                    return_loss=False, rescale=True, **data_gpu)
            results[idx] = result

            batch_size = runner.world_size
            if runner.rank == 0:
                for _ in range(batch_size):
                    prog_bar.update()

        if runner.rank == 0:
            print('\n')
            dist.barrier()
            for i in range(1, runner.world_size):
                tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(i))
                tmp_results = mmcv.load(tmp_file)
                for idx in range(i, len(results), runner.world_size):
                    results[idx] = tmp_results[idx]
                os.remove(tmp_file)
            self.evaluate(runner, results)
        else:
            tmp_file = osp.join(runner.work_dir,
                                'temp_{}.pkl'.format(runner.rank))
            mmcv.dump(results, tmp_file)
            dist.barrier()
        dist.barrier()

    def evaluate(self):
        raise NotImplementedError


class DistEvalHook3D(Hook):

    def __init__(self, dataset, interval=2, dataset2=None, dataset3=None):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = obj_from_dict(dataset, datasets,
                                         {'test_mode': True})
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))

        if dataset2 is not None:
            if isinstance(dataset2, Dataset):
                self.dataset2 = dataset2
            elif isinstance(dataset2, dict):
                self.dataset2 = obj_from_dict(dataset2, datasets,
                                            {'test_mode': True})
        else:
            self.dataset2 = None
        
        if dataset3 is not None:
            if isinstance(dataset3, Dataset):
                self.dataset3 = dataset3
            elif isinstance(dataset3, dict):
                self.dataset3 = obj_from_dict(dataset3, datasets,
                                            {'test_mode': True})
        else:
            self.dataset3 = None

        self.interval = interval

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()
        results = [None for _ in range(len(self.dataset))]
        if runner.rank == 0:
            prog_bar = mmcv.ProgressBar(len(self.dataset))
        for idx in range(runner.rank, len(self.dataset), runner.world_size):
            data = self.dataset[idx]
            data_gpu = scatter(
                collate([data], samples_per_gpu=1),
                [torch.cuda.current_device()])[0]
            # compute output
            with torch.no_grad():
                result = runner.model(
                    return_loss=False, rescale=True, **data_gpu)
            results[idx] = result

            batch_size = runner.world_size
            if runner.rank == 0:
                for _ in range(batch_size):
                    prog_bar.update()
        
        if runner.rank == 0:
            print('\n')
            dist.barrier()
            for i in range(1, runner.world_size):
                tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(i))
                tmp_results = mmcv.load(tmp_file)
                for idx in range(i, len(results), runner.world_size):
                    results[idx] = tmp_results[idx]
                os.remove(tmp_file)
            self.evaluate(runner, results)
        else:
            tmp_file = osp.join(runner.work_dir,
                                'temp_{}.pkl'.format(runner.rank))
            mmcv.dump(results, tmp_file)
            dist.barrier()
        dist.barrier()

    def evaluate(self):
        raise NotImplementedError


class DistEvalmAPHook(DistEvalHook):

    def evaluate(self, runner, results):
        gt_bboxes = []
        gt_labels = []
        gt_ignore = [] if self.dataset.with_crowd else None
        for i in range(len(self.dataset)):
            ann = self.dataset.get_ann_info(i)
            bboxes = ann['bboxes']
            labels = ann['labels']
            if gt_ignore is not None:
                ignore = np.concatenate([
                    np.zeros(bboxes.shape[0], dtype=np.bool),
                    np.ones(ann['bboxes_ignore'].shape[0], dtype=np.bool)
                ])
                gt_ignore.append(ignore)
                bboxes = np.vstack([bboxes, ann['bboxes_ignore']])
                labels = np.concatenate([labels, ann['labels_ignore']])
            gt_bboxes.append(bboxes)
            gt_labels.append(labels)
        # If the dataset is VOC2007, then use 11 points mAP evaluation.
        if hasattr(self.dataset, 'year') and self.dataset.year == 2007:
            ds_name = 'voc07'
        else:
            ds_name = self.dataset.CLASSES
        mean_ap, eval_results = eval_map(
            results,
            gt_bboxes,
            gt_labels,
            gt_ignore=gt_ignore,
            scale_ranges=None,
            iou_thr=0.5,
            dataset=ds_name,
            print_summary=True)
        runner.log_buffer.output['mAP'] = mean_ap
        runner.log_buffer.ready = True


class CocoDistEvalRecallHook(DistEvalHook):

    def __init__(self,
                 dataset,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05)):
        super(CocoDistEvalRecallHook, self).__init__(dataset)
        self.proposal_nums = np.array(proposal_nums, dtype=np.int32)
        self.iou_thrs = np.array(iou_thrs, dtype=np.float32)

    def evaluate(self, runner, results):
        # the official coco evaluation is too slow, here we use our own
        # implementation instead, which may get slightly different results
        ar = fast_eval_recall(results, self.dataset.coco, self.proposal_nums,
                              self.iou_thrs)
        for i, num in enumerate(self.proposal_nums):
            runner.log_buffer.output['AR@{}'.format(num)] = ar[i]
        runner.log_buffer.ready = True


class CocoDistEvalmAPHook(DistEvalHook):

    def evaluate(self, runner, results):
        no_prediction = True
        for result in results:
            if len(result[0][0]) > 0:
                no_prediction = False
                break
        
        if no_prediction is False:
            tmp_file = osp.join(runner.work_dir, 'temp_0.json')
            results2json(self.dataset, results, tmp_file)

            res_types = ['bbox',
                        'segm'] if runner.model.module.with_mask else ['bbox']
            cocoGt = self.dataset.coco
            cocoDt = cocoGt.loadRes(tmp_file)
            imgIds = cocoGt.getImgIds()
            for res_type in res_types:
                iou_type = res_type
                cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
                cocoEval.params.imgIds = imgIds
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                metrics = ['mAP', 'mAP_0.05', 'mAP_0.1', 'mAP_0.15', 'mAP_0.2', 
                'mAP_0.25', 'mAP_0.3', 'mAP_0.35', 'mAP_0.4', 'mAP_0.45', 'mAP_0.5', 
                'mAP_0.55', 'mAP_0.6', 'mAP_0.65', 'mAP_0.7', 'mAP_0.75', 'mAP_0.8',
                'mAP_0.85', 'mAP_0.9', 'mAP_0.95', 'mAP_small', 'mAP_medium', 
                'mAP_large', 'mAR_max_dets_1_all', 'mAR_max_dets_10_all', 'mAR_max_dets_100_all',
                'mAR_max_dets_100_small', 'mAR_max_dets_100_medium', 'mAR_max_dets_100_large']
                for i in range(len(metrics)):
                    key = '{}_{}'.format(res_type, metrics[i])
                    val = float('{:.3f}'.format(cocoEval.stats[i]))
                    runner.log_buffer.output[key] = val
                runner.log_buffer.output['{}_mAP_copypaste'.format(res_type)] = (
                    '{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    '{ap[4]:.3f} {ap[5]:.3f}').format(ap=cocoEval.stats[:6])
            runner.log_buffer.ready = True
            os.remove(tmp_file)

class CocoDistEvalmAPHook3D(DistEvalHook3D):

    def evaluate(self, runner, results):
        no_prediction = True
        for result in results:
            if len(result[0]) > 0 and len(result[0][0]) > 0:
                no_prediction = False
                break
        
        if no_prediction is False:
            tmp_file = osp.join(runner.work_dir, 'temp_0.json')

            # TODO: temporarily disable segmentation
            res_types = ['bbox']
            # res_types = ['bbox',
            #             'segm'] if runner.model.module.with_mask else ['bbox']

            cocoGt = COCO(self.dataset.ann_file_volume)


            full_filename_to_id = dict()
            for img_id in cocoGt.getImgIds():
                full_filename_to_id[cocoGt.loadImgs([img_id])[0]['file_name']] = img_id

            if runner.model.module.with_mask:
                result = results2json3D(self.dataset, results, tmp_file, full_filename_to_id)
                cocoDt = cocoGt.loadRes3D(result)
            else:    
                results2json3D(self.dataset, results, tmp_file, full_filename_to_id)
                cocoDt = cocoGt.loadRes(tmp_file)
            imgIds = cocoGt.getImgIds()
            for res_type in res_types:
                iou_type = res_type
                cocoEval = COCOeval(cocoGt, cocoDt, iou_type, is3D=True)
                cocoEval.params.imgIds = imgIds
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                metrics = ['mAP', 'mAP_0.05', 'mAP_0.1', 'mAP_0.15', 'mAP_0.2', 
                'mAP_0.25', 'mAP_0.3', 'mAP_0.35', 'mAP_0.4', 'mAP_0.45', 'mAP_0.5', 
                'mAP_0.55', 'mAP_0.6', 'mAP_0.65', 'mAP_0.7', 'mAP_0.75', 'mAP_0.8',
                'mAP_0.85', 'mAP_0.9', 'mAP_0.95', 'mAP_small', 'mAP_medium', 
                'mAP_large', 'mAR_max_dets_1_all', 'mAR_max_dets_10_all', 'mAR_max_dets_100_all',
                'mAR_max_dets_100_small', 'mAR_max_dets_100_medium', 'mAR_max_dets_100_large']

                for i in range(len(metrics)):
                    key = '{}_{}'.format(res_type, metrics[i])
                    val = float('{:.3f}'.format(cocoEval.stats[i]))
                    runner.log_buffer.output[key] = val
                runner.log_buffer.output['{}_mAP_copypaste'.format(res_type)] = (
                    '{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    '{ap[4]:.3f} {ap[5]:.3f}').format(ap=cocoEval.stats[:6])
            runner.log_buffer.ready = True
            os.remove(tmp_file)