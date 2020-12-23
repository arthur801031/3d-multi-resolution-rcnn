import numpy as np
from pycocotools_local.coco import *
import os.path as osp
from .utils import to_tensor, random_scale
from mmcv.parallel import DataContainer as DC
import mmcv

from .custom import CustomDataset


class CocoDatasetRGB2(CustomDataset):

    CLASSES = ('microbleed', 'full_bounding_box')

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info, self.with_mask)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, ann_info, with_mask=True):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        slices_ann_info = {'r': [], 'g': [], 'b': []}

        for info in ann_info:
            if info['slice_label'] == 'r':
                slices_ann_info['r'].append(info)

            elif info['slice_label'] == 'g':
                slices_ann_info['g'].append(info)

            elif info['slice_label'] == 'b':
                slices_ann_info['b'].append(info)

        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.
        if with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []

        for key in slices_ann_info:
            cur_ann_info = slices_ann_info[key]
            cur_slice_bboxes = []
            cur_slice_labels = []
            cur_slice_bboxes_ignore = []
            cur_masks = []
            cur_mask_polys = []
            cur_poly_lens = []

            for i, ann in enumerate(cur_ann_info):
                if ann.get('ignore', False):
                    continue
                x1, y1, w, h = ann['bbox']
                if ann['area'] <= 0 or w < 1 or h < 1:
                    continue
                bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
                if ann['iscrowd']:
                    cur_slice_bboxes_ignore.append(bbox)
                else:
                    cur_slice_bboxes.append(bbox)
                    cur_slice_labels.append(self.cat2label[ann['category_id']])
                if with_mask:
                    cur_masks.append(self.coco.annToMask(ann))
                    mask_polys = [
                        p for p in ann['segmentation'] if len(p) >= 6
                    ]  # valid polygons have >= 3 points (6 coordinates)
                    poly_lens = [len(p) for p in mask_polys]
                    cur_mask_polys.append(mask_polys)
                    cur_poly_lens.extend(poly_lens)
            if cur_slice_bboxes:
                cur_slice_bboxes = np.array(cur_slice_bboxes, dtype=np.float32)
                cur_slice_labels = np.array(cur_slice_labels, dtype=np.int64)
            else:
                cur_slice_bboxes = np.zeros((0, 4), dtype=np.float32)
                cur_slice_labels = np.array([], dtype=np.int64)

            if cur_slice_bboxes_ignore:
                cur_slice_bboxes_ignore = np.array(cur_slice_bboxes_ignore, dtype=np.float32)
            else:
                cur_slice_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
            
            gt_bboxes.append(cur_slice_bboxes)
            gt_labels.append(cur_slice_labels)
            gt_bboxes_ignore.append(cur_slice_bboxes_ignore)
            gt_masks.append(cur_masks)
            gt_mask_polys.append(cur_mask_polys)
            gt_poly_lens.append(cur_poly_lens)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)

        if with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens
        return ann

    def insert_to_dict(self, data, key, tensors):
        if key in data:
            data[key].append(tensors)
        else:
            data[key] = [tensors]

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        # load image
        orig_img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None

        ann = self.get_ann_info(idx)
        gt_bboxes_list = ann['bboxes']
        gt_labels_list = ann['labels']
        # if self.with_crowd:
        gt_bboxes_ignore_list = ann['bboxes_ignore']
        
        gt_masks_list = ann['masks']

         # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False

        data = None
        for gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks in zip(gt_bboxes_list, gt_labels_list, gt_bboxes_ignore_list, gt_masks_list):
            # skip the image if there is no valid gt bbox
            if len(gt_bboxes) == 0:
                return None

            # extra augmentation
            if self.extra_aug is not None:
                img, gt_bboxes, gt_labels = self.extra_aug(orig_img, gt_bboxes,
                                                        gt_labels)
            else:
                img = orig_img
  
            # randomly sample a scale
            img_scale = random_scale(self.img_scales, self.multiscale_mode)
            img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
            img = img.copy()
            if self.with_seg:
                gt_seg = mmcv.imread(
                    osp.join(self.seg_prefix, img_info['file_name'].replace(
                        'jpg', 'png')),
                    flag='unchanged')
                gt_seg = self.seg_transform(gt_seg.squeeze(), img_scale, flip)
                gt_seg = mmcv.imrescale(
                    gt_seg, self.seg_scale_factor, interpolation='nearest')
                gt_seg = gt_seg[None, ...]
            if self.proposals is not None:
                proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                                flip)
                proposals = np.hstack(
                    [proposals, scores]) if scores is not None else proposals
            gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                            flip)
            if self.with_crowd:
                gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                    scale_factor, flip)
            if self.with_mask:
                gt_masks = self.mask_transform(gt_masks, pad_shape,
                                            scale_factor, flip)
            
            if data is None:
                ori_shape = (img_info['height'], img_info['width'], 3)
                img_meta = dict(
                    ori_shape=ori_shape,
                    img_shape=img_shape,
                    pad_shape=pad_shape,
                    scale_factor=scale_factor,
                    flip=flip,
                    image_id=img_info['id'])

                data = dict(
                    img=DC(to_tensor(img), stack=True),
                    img_meta=DC(img_meta, cpu_only=True))
            
            self.insert_to_dict(data, 'gt_bboxes', DC(to_tensor(gt_bboxes)))

            if self.proposals is not None:
                self.insert_to_dict(data, 'proposals', DC(to_tensor(proposals)))
            if self.with_label:
                self.insert_to_dict(data, 'gt_labels', DC(to_tensor(gt_labels)))
            if self.with_crowd:
                self.insert_to_dict(data, 'gt_bboxes_ignore', DC(to_tensor(gt_bboxes_ignore)))
            if self.with_mask:
                self.insert_to_dict(data, 'gt_masks', DC(gt_masks, cpu_only=True))
            if self.with_seg:
                self.insert_to_dict(data, 'gt_semantic_seg', DC(to_tensor(gt_seg), stack=True))

        return data