import mmcv
import numpy as np
from pycocotools_local.coco import *
import os.path as osp
from .utils import to_tensor, random_scale
from mmcv.parallel import DataContainer as DC

from .custom import CustomDataset
from .forkedpdb import ForkedPdb
from skimage.transform import resize

class Coco3D2ScalesDataset(CustomDataset):

    CLASSES = ('microbleed')

    def load_annotations(self, ann_file, ann_file_2=None):
        if ann_file_2 is None:
            ann_file_2 = ann_file

        self.coco = COCO(ann_file)
        self.coco_2 = COCO(ann_file_2)

        self.cat_ids = self.coco.getCatIds()
        self.cat_ids_2 = self.coco_2.getCatIds()

        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.cat2label_2 = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids_2)
        }

        self.img_ids = self.coco.getImgIds()
        self.img_ids_2 = self.coco_2.getImgIds()

        img_infos, img_infos_2 = [], []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        for i in self.img_ids_2:
            info = self.coco_2.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos_2.append(info)
        return img_infos, img_infos_2

    def get_ann_infos(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)

        img_id_2 = self.img_infos_2[idx]['id']
        ann_ids_2 = self.coco_2.getAnnIds(imgIds=[img_id_2])
        ann_info_2 = self.coco_2.loadAnns(ann_ids_2)

        # return self._parse_ann_info(ann_info, False), self._parse_ann_info(ann_info_2, self.with_mask)
        ann = self._parse_ann_info(ann_info, idx, self.with_mask, isOriginalScale=True)
        ann_2 = self._parse_ann_info(ann_info_2, idx, self.with_mask, isOriginalScale=False)
        return ann, ann_2

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)

        valid_inds_2 = []
        ids_with_ann_2 = set(_['image_id'] for _ in self.coco_2.anns.values())
        for i, img_info in enumerate(self.img_infos_2):
            if self.img_ids_2[i] not in ids_with_ann_2:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds_2.append(i)
        return valid_inds, valid_inds_2

    def _parse_ann_info(self, ann_info, idx, with_mask=True, isOriginalScale=False):
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
                # error: ValueError: Cannot load file containing pickled data when allow_pickle=False
                if isOriginalScale:
                    if self.load_mask_from_memory and self.seg_masks_memory_isloaded[idx] is False:
                        mask = np.load(ann['segmentation'], allow_pickle=True)
                        mask[mask != ann['segmentation_label']] = 0
                        mask[mask == ann['segmentation_label']] = 1
                        self.seg_masks_memory[idx][i, :, :, :] = mask
                    elif self.load_mask_from_memory and self.seg_masks_memory_isloaded[idx] is True:
                        mask = self.seg_masks_memory[idx][i, :, :, :]
                    elif self.load_mask_from_memory is None:
                        mask = np.load(ann['segmentation'], allow_pickle=True)
                        mask[mask != ann['segmentation_label']] = 0
                        mask[mask == ann['segmentation_label']] = 1
                else:
                    mask = None
                gt_masks.append(mask)

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

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)

        if with_mask:
            ann['masks'] = gt_masks
        
        if self.load_mask_from_memory:
            self.seg_masks_memory_isloaded[idx] = True
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
        scale_factor = 1.0
        flip = False
        img_info = self.img_infos[idx]
        img_info_2 = self.img_infos_2[idx]

        # ForkedPdb().set_trace()

        # ensure it's the same image but at different scales
        assert img_info['filename'] == img_info_2['filename']

        img_file_path = osp.join(self.img_prefix, img_info['filename'])
        img_file_path_2 = osp.join(self.img_prefix_2, img_info_2['filename'])
        orig_img = np.load(img_file_path, allow_pickle=True)
        orig_img_2 = np.load(img_file_path_2, allow_pickle=True)

        upscale_factor = orig_img_2.shape[0]/orig_img.shape[0]
        total_num_slices = orig_img.shape[2]

        ann, ann_2 = self.get_ann_infos(idx)
        
        gt_bboxes = ann['bboxes']
        gt_bboxes_2 = ann_2['bboxes']
        gt_labels = ann['labels']
        gt_labels_2 = ann_2['labels']
        if 'masks' not in ann and 'masks' in ann_2:
            gt_masks = ann_2['masks']
            gt_masks_2 = None
        elif 'masks' in ann and 'masks' in ann_2:
            gt_masks = ann['masks']
            gt_masks_2 = ann_2['masks']
        else:
            gt_masks = None
            gt_masks_2 = None

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0 or len(gt_bboxes_2) == 0:
            return None

        # extra augmentation
        if self.extra_aug is not None:
            # orig_img, gt_bboxes, gt_labels, _ = self.extra_aug(orig_img, gt_bboxes, gt_labels, None)
            # img_scale = (orig_img.shape[0], orig_img.shape[1]) # disable scaling...
            # orig_img_2, gt_bboxes_2, gt_labels_2, gt_masks = self.extra_aug(orig_img_2, gt_bboxes_2, gt_labels_2, gt_masks)
            # img_scale_2 = (orig_img_2.shape[0], orig_img_2.shape[1]) # disable scaling...

            orig_img, gt_bboxes, gt_labels, gt_masks = self.extra_aug(orig_img, gt_bboxes, gt_labels, gt_masks)
            img_scale = (orig_img.shape[0], orig_img.shape[1]) # disable scaling...
            # upscale original scale patch so that during training original scale data and upscaled data are the same 
            # patch but with different scale
            orig_img_2 = resize(orig_img, (orig_img.shape[0]*upscale_factor, orig_img.shape[1]*upscale_factor, orig_img.shape[2]*upscale_factor))

            # TODO: Better way to toggle on/off gt_masks_2. Currently it is turned off.
            # if gt_masks is not None:
            #     gt_masks_2 = []
            #     for cur_mask in gt_masks:
            #         gt_masks_2.append(resize(cur_mask, (cur_mask.shape[0]*upscale_factor, cur_mask.shape[1]*upscale_factor, cur_mask.shape[2]*upscale_factor)))
            #     gt_masks_2 = np.array(gt_masks_2)
            gt_masks_2 = None

            gt_bboxes_2 = gt_bboxes * upscale_factor
            gt_labels_2 = gt_labels

            # original code
            # orig_img_2, gt_bboxes_2, gt_labels_2, gt_masks_2 = self.extra_aug(orig_img_2, gt_bboxes_2, gt_labels_2, gt_masks_2)
            img_scale_2 = (orig_img_2.shape[0], orig_img_2.shape[1]) # disable scaling...
        else:
            # randomly sample a scale
            img_scale = random_scale(self.img_scales, self.multiscale_mode)
            img_scale_2 = random_scale(self.img_scales_2, self.multiscale_mode)
   
        total_num_slices = orig_img.shape[2]
        total_num_slices_2 = orig_img_2.shape[2]
        data = None
        for cur_slice in range(total_num_slices):
            img = orig_img[:,:,cur_slice]
            # convert Greyscale to RGB
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

            img, img_shape, pad_shape, _ = self.img_transform(
                img, img_scale, flip, keep_ratio=self.resize_keep_ratio)

            img = img.copy()
            if data is None:
                ori_shape = (img.shape[1], img.shape[2], 3)
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

        first_iter = True
        for cur_slice in range(total_num_slices_2):
            img = orig_img_2[:,:,cur_slice]
            # convert Greyscale to RGB
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

            img, img_shape, pad_shape, _ = self.img_transform(
                img, img_scale_2, flip, keep_ratio=self.resize_keep_ratio)

            img = img.copy()
            if first_iter:
                ori_shape = (img.shape[1], img.shape[2], 3)
                img_shape_2 = (*img_shape, total_num_slices_2)
                pad_shape = (*pad_shape, total_num_slices_2)
                img_meta = dict(
                    ori_shape=ori_shape,
                    img_shape=img_shape_2,
                    pad_shape=pad_shape,
                    scale_factor=scale_factor,
                    flip=flip,
                    image_id=img_info_2['id'])
                data['img_meta_2'] = DC(img_meta, cpu_only=True)
                first_iter = False
            self.insert_to_dict(data, 'imgs_2', img)

        gt_bboxes = self.bbox_transform(gt_bboxes, (*img_shape, total_num_slices), scale_factor, flip)
        gt_bboxes_2 = self.bbox_transform(gt_bboxes_2, (*img_shape_2, total_num_slices_2), scale_factor, flip)
        data['gt_bboxes'] = DC(to_tensor(gt_bboxes))
        data['gt_bboxes_2'] = DC(to_tensor(gt_bboxes_2))
        
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
            data['gt_labels_2'] = DC(to_tensor(gt_labels_2))
        if gt_masks is not None:
            gt_masks = self.mask_transform(gt_masks, pad_shape, scale_factor, flip, is3D=True)
            gt_masks = gt_masks.transpose(0,3,1,2)
            data['gt_masks'] = DC(to_tensor(gt_masks.astype(np.uint8)), cpu_only=True)

        if gt_masks_2 is not None:
            gt_masks_2 = self.mask_transform(gt_masks_2, pad_shape, scale_factor, flip, is3D=True)
            gt_masks_2 = gt_masks_2.transpose(0,3,1,2)
            data['gt_masks_2'] = DC(to_tensor(gt_masks_2.astype(np.uint8)), cpu_only=True)

        imgs = np.array(data['imgs'])
        imgs = imgs.transpose(1, 0, 2, 3)
        data['imgs'] = DC(to_tensor(imgs), stack=True)

        imgs_2 = np.array(data['imgs_2'])
        imgs_2 = imgs_2.transpose(1, 0, 2, 3)
        data['imgs_2'] = DC(to_tensor(imgs_2), stack=True)

        if self.with_precomp_proposals:
            # load unsupervised learning's proposals
            pp = np.load('precomputed-proposals.pickle', allow_pickle=True)
            pp_2 = np.load('precomputed-proposals1.5.pickle', allow_pickle=True)
            pp = pp[img_info['filename'].split('.')[0]]
            pp_2 = pp_2[img_info_2['filename'].split('.')[0]]
            data['pp'] = DC(to_tensor(pp))
            data['pp_2'] = DC(to_tensor(pp_2))

        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info = self.img_infos[idx]

        # find corresponding img_info for 1.5x dataset
        index_2 = -1
        for i, cur_info in enumerate(self.img_infos_2):
            if cur_info['filename'] == img_info['filename']:
                index_2 = i
    
        img_info_2 = self.img_infos_2[index_2]
        patient_imgs = np.load((osp.join(self.img_prefix, img_info['filename'])), allow_pickle=True)
        patient_imgs_2 = np.load((osp.join(self.img_prefix_2, img_info_2['filename'])), allow_pickle=True)

        # scale_factor = 1.0 / (img_info_2['width'] / img_info['width']) # scale up to img_info_2's resolution
        scale_factor = 1 # remain the same 
        scale_factor_2 = (patient_imgs_2.shape[0] / patient_imgs.shape[0]) # scale up to img_info_2's resolution
        total_num_slices = patient_imgs.shape[2]
        total_num_slices_2 = patient_imgs_2.shape[2]
    
        def prepare_single(img, scale, flip, img_info, cur_total_num_slices, scale_factor, proposal=None):
            _img, img_shape, pad_shape, _ = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)

            # old code without resizing depth
            # _img, img_shape, pad_shape, scale_factor = self.img_transform(
            #     img, scale, flip, keep_ratio=self.resize_keep_ratio)
            img_shape = (*img_shape, cur_total_num_slices)
            pad_shape = (*pad_shape, cur_total_num_slices)
            _img_meta = dict(
                ori_shape=(_img.shape[1], _img.shape[2], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
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


        proposal = None
        imgs = []
        img_metas = []
        for cur_slice in range(total_num_slices):
            img = patient_imgs[:,:,cur_slice]
            # convert Greyscale to RGB
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

            for scale in self.img_scales:
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, False, img_info, total_num_slices, scale_factor)
                imgs.append(_img)
                if len(img_metas) == 0:
                    img_metas.append(DC(_img_meta, cpu_only=True))

        imgs_2 = []
        img_metas_2 = []
        for cur_slice in range(total_num_slices_2):
            img_2 = patient_imgs_2[:,:,cur_slice]
            # convert Greyscale to RGB
            img_2 = np.repeat(img_2[:, :, np.newaxis], 3, axis=2)

            for scale in self.img_scales_2:
                _img, _img_meta, _proposal = prepare_single(
                    img_2, scale, False, img_info_2, total_num_slices_2, scale_factor_2)
                imgs_2.append(_img)
                if len(img_metas_2) == 0:
                    img_metas_2.append(DC(_img_meta, cpu_only=True))
                
        
        imgs = np.array(imgs)
        imgs_2 = np.array(imgs_2)

        imgs = np.transpose(imgs, (1, 0, 2, 3))
        imgs_2 = np.transpose(imgs_2, (1, 0, 2, 3))

        assert imgs.shape[0] == 3 and imgs_2.shape[0] == 3# make sure [0] is the number of channels
        data = dict(imgs=to_tensor(imgs), img_meta=img_metas, imgs_2=to_tensor(imgs_2), img_meta_2=img_metas_2)

        if self.with_precomp_proposals:
            # load unsupervised learning's proposals
            pp = np.load('precomputed-proposals.pickle', allow_pickle=True)
            pp_2 = np.load('precomputed-proposals1.5.pickle', allow_pickle=True)
            pp = pp[img_info['filename'].split('.')[0]]
            pp_2 = pp_2[img_info_2['filename'].split('.')[0]]
            data['pp'] = to_tensor(pp)
            data['pp_2'] = to_tensor(pp_2)

        return data