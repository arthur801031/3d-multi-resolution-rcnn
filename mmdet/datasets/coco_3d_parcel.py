import mmcv
import numpy as np
from pycocotools_local.coco import *
import os.path as osp
from .utils import to_tensor, random_scale
from mmcv.parallel import DataContainer as DC

from .custom import CustomDataset
from .forkedpdb import ForkedPdb
from PIL import Image
import cv2
from skimage.transform import resize

class Coco3DParcelDataset(CustomDataset):

    CLASSES = ('microbleed')

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
        gt_bboxes = []
        gt_labels = []
        gt_bregions = []
        gt_bboxes_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.
        if with_mask:
            gt_masks = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h, z1, depth = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1 or depth < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1, z1, z1 + depth - 1]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
            
            if with_mask:
                mask = np.load(ann['segmentation'])
                mask[mask != ann['segmentation_label']] = 0
                mask[mask == ann['segmentation_label']] = 1
                gt_masks.append(mask)
            if ann.get('brain_region') is not None:
                gt_bregions.append(ann['brain_region'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 6), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 6), dtype=np.float32)

        gt_bregions = np.array(gt_bregions, dtype=np.int64)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore, gt_bregions=gt_bregions)

        if with_mask:
            ann['masks'] = gt_masks
        return ann

    def find_indices_of_slice(self, gt_bboxes, slice_num):
        indices = []
        for i in range(len(gt_bboxes)):
            _, _, _, _, zmin, zmax = gt_bboxes[i]
            if zmin <= slice_num and slice_num <= zmax:
                indices.append(i)
        return indices

    def insert_to_dict(self, data, key, tensors):
        if key in data:
            data[key].append(tensors)
        else:
            data[key] = [tensors]

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]

        img_file_path = osp.join(self.img_prefix, img_info['filename'])
        img_np = np.load(img_file_path)

        total_num_slices = img_np.shape[2]

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

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']
        gt_bregions = ann['gt_bregions']
        if 'masks' in ann:
            gt_masks = ann['masks']
        else:
            gt_masks = None

        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None

        # randomly sample a scale
        img_scale = random_scale(self.img_scales, self.multiscale_mode)

        # extra augmentation
        if self.extra_aug is not None:
            orig_img, gt_bboxes, gt_labels, gt_masks = self.extra_aug(img_np, gt_bboxes, gt_labels, gt_masks)
            # img_scale = (orig_img.shape[0], orig_img.shape[1]) # disable scaling...
        else:
            orig_img = img_np

        scale_factor = img_scale[0]/orig_img.shape[0]
        if scale_factor > 1.0:
            ForkedPdb().set_trace()
            #  orig_img = resize(orig_img, (orig_img.shape[0] * scale_factor, orig_img.shape[1] * scale_factor, orig_img.shape[2] * scale_factor))
            # orig_img = zoom(orig_img, (scale_factor, scale_factor, scale_factor))
            orig_img = 255 * resize(orig_img.astype(np.uint8), (orig_img.shape[0] * scale_factor, orig_img.shape[1] * scale_factor, orig_img.shape[2] * scale_factor))
            orig_img = orig_img.astype(np.uint8)

        total_num_slices = orig_img.shape[2]
        data = None
        for cur_slice in range(total_num_slices):
            img = orig_img[:,:,cur_slice]
            img = Image.fromarray(img).convert('RGB')
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            img, img_shape, pad_shape, _ = self.img_transform(
                img, img_scale, flip, keep_ratio=self.resize_keep_ratio)

            img = img.copy()
            if self.with_seg:
                gt_seg = mmcv.imread(
                    osp.join(self.seg_prefix, img_info['file_name'].replace(
                        'jpg', 'png')),
                    flag='unchanged')
                # TODO: implement flip for segmentation....
                gt_seg = self.seg_transform(gt_seg.squeeze(), img_scale, flip)
                gt_seg = mmcv.imrescale(
                    gt_seg, self.seg_scale_factor, interpolation='nearest')
                gt_seg = gt_seg[None, ...]
            if self.proposals is not None:
                proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                                flip)
                proposals = np.hstack(
                    [proposals, scores]) if scores is not None else proposals

            if self.with_crowd:
                cur_bboxes_ignore = self.bbox_transform(cur_bboxes_ignore, img_shape,
                                                    scale_factor, flip)

            if data is None:
                ori_shape = (img_info['height'], img_info['width'], 3)
                img_shape = (*img_shape, total_num_slices)
                pad_shape = (*pad_shape, total_num_slices)
                img_meta = dict(
                    ori_shape=ori_shape,
                    img_shape=img_shape,
                    pad_shape=pad_shape,
                    scale_factor=scale_factor,
                    flip=flip,
                    image_id=img_info['id'])
                data = dict(img_meta=DC(img_meta, cpu_only=True))

            self.insert_to_dict(data, 'imgs', img)

        gt_bboxes = self.bbox_transform(gt_bboxes, (*img_shape, total_num_slices), scale_factor, flip)
        data['gt_bboxes'] = DC(to_tensor(gt_bboxes))
        
        data['gt_bregions'] = DC(to_tensor(gt_bregions))
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        if self.with_mask:
            gt_masks = self.mask_transform(gt_masks, pad_shape, scale_factor, flip, is3D=True)
            gt_masks = gt_masks.transpose(0,3,1,2)
            data['gt_masks'] = DC(to_tensor(gt_masks.astype(np.uint8)), cpu_only=True)
        imgs = np.array(data['imgs'])
        imgs = imgs.transpose(1, 0, 2, 3)
        data['imgs'] = DC(to_tensor(imgs), stack=True)

        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info = self.img_infos[idx]
        patient_imgs = np.load((osp.join(self.img_prefix, img_info['filename'])))

        scale_factor = self.img_scales[0][0]/patient_imgs.shape[0]
        if scale_factor > 1.0:
            ForkedPdb().set_trace()
            #  patient_imgs = resize(patient_imgs, (patient_imgs.shape[0] * scale_factor, patient_imgs.shape[1] * scale_factor, patient_imgs.shape[2] * scale_factor))
            # patient_imgs = zoom(patient_imgs, (scale_factor, scale_factor, scale_factor))
            patient_imgs = 255 * resize(patient_imgs.astype(np.uint8), (patient_imgs.shape[0] * scale_factor, patient_imgs.shape[1] * scale_factor, patient_imgs.shape[2] * scale_factor))
            patient_imgs = patient_imgs.astype(np.uint8)
        total_num_slices = patient_imgs.shape[2]
    
        def prepare_single(img, scale, flip, total_num_slices, proposal=None):
            _img, img_shape, pad_shape, _ = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)

            # old code without resizing depth
            # _img, img_shape, pad_shape, scale_factor = self.img_transform(
            #     img, scale, flip, keep_ratio=self.resize_keep_ratio)
            img_shape = (*img_shape, total_num_slices)
            pad_shape = (*pad_shape, total_num_slices)
            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                filename=img_info['filename'].split('.')[0],
                flip=flip)
            if proposal is not None:
                breakpoint()
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack(
                    [_proposal, score]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img, _img_meta, _proposal
        
        if self.proposals is not None:
            breakpoint()
            proposal = self.proposals[idx][:self.num_max_proposals]
            if not (proposal.shape[1] == 4 or proposal.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposal.shape))
        else:
            proposal = None

        imgs = []
        img_metas = []
        proposals = []
        for cur_slice in range(total_num_slices):
            img = patient_imgs[:,:,cur_slice]
            img = Image.fromarray(img).convert('RGB')
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            for scale in self.img_scales:
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, False, total_num_slices, proposal)
                imgs.append(_img)
                if len(img_metas) == 0:
                    img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)
                if self.flip_ratio > 0:
                    breakpoint()
                    _img, _img_meta, _proposal = prepare_single(
                        img, scale, True, total_num_slices, proposal)
                    imgs.append(_img)
                    if len(img_metas) == 1:
                        img_metas.append(DC(_img_meta, cpu_only=True))
                    proposals.append(_proposal)
        
        imgs = np.array(imgs)
        imgs = np.transpose(imgs, (1, 0, 2, 3))
        assert imgs.shape[0] == 3 # make sure [0] is the number of channels
        data = dict(imgs=to_tensor(imgs), img_meta=img_metas)
        if self.proposals is not None:
            data['proposals'] = proposals
        return data