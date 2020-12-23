from __future__ import division

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import (AnchorGenerator3D, anchor_target, delta2bbox,
                        multi_apply, weighted_cross_entropy, weighted_smoothl1,
                        weighted_binary_cross_entropy,
                        weighted_sigmoid_focal_loss, multiclass_nms)
from mmdet.core.anchor.anchor_target import get_anchor_inside_flags
from ..registry import HEADS
from mmdet.core import tensor2img3D
import mmcv
import random
import cv2
import torch.nn.functional as F

@HEADS.register_module
class AnchorHead3D(nn.Module):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).

    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        anchor_scales (Iterable): Anchor scales.
        anchor_ratios (Iterable): Anchor aspect ratios.
        anchor_strides (Iterable): Anchor strides.
        anchor_base_sizes (Iterable): Anchor base sizes.
        target_means (Iterable): Mean values of regression targets.
        target_stds (Iterable): Std values of regression targets.
        use_sigmoid_cls (bool): Whether to use sigmoid loss for classification.
            (softmax by default)
        use_focal_loss (bool): Whether to use focal loss for classification.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 anchor_scales=[8, 16, 32],
                 anchor_depth_scales=[3],
                 anchor_ratios=[0.5, 1.0, 2.0],
                 anchor_strides=[4, 8, 16, 32, 64],
                 anchor_strides_depth=[2, 4, 8, 16, 32],
                 anchor_base_sizes=None,
                 target_means=(.0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0),
                 use_sigmoid_cls=False,
                 use_focal_loss=False):
        super(AnchorHead3D, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.anchor_scales = anchor_scales
        self.anchor_depth_scales = anchor_depth_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.anchor_base_sizes = list(
            anchor_strides) if anchor_base_sizes is None else anchor_base_sizes
        self.anchor_strides_depth = anchor_strides_depth
        self.target_means = target_means
        self.target_stds = target_stds
        self.use_sigmoid_cls = use_sigmoid_cls
        self.use_focal_loss = use_focal_loss
        self.pos_indices = None
        self.pos_indices_test = None

        self.anchor_generators = []
        for anchor_base, anchor_depth_base in zip(self.anchor_base_sizes, self.anchor_strides_depth):
            self.anchor_generators.append(
                AnchorGenerator3D(anchor_base, anchor_scales, anchor_depth_scales, anchor_ratios, anchor_depth_base))

        self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes - 1
        else:
            self.cls_out_channels = self.num_classes

        self._init_layers()

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, img_metas):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = []
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i], self.anchor_strides_depth[i])
            multi_level_anchors.append(anchors)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                anchor_depth_stride = self.anchor_strides_depth[i]
                feat_z, feat_h, feat_w = featmap_sizes[i]
                h, w, _, z = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                valid_feat_z = min(int(np.ceil(z / anchor_depth_stride)), feat_z)
                flags = self.anchor_generators[i].valid_flags(
                    (feat_z, feat_h, feat_w), (valid_feat_z, valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_weights, anchors, num_total_samples, cfg, level, gt_bboxes, iteration):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        # 3D images
        cls_score = cls_score.permute(0, 3, 4, 2, 1).reshape(-1, self.cls_out_channels)
        # debug only...
        # self.visualize_anchors_across_levels(anchors, gt_bboxes, labels, label_weights, iteration=iteration, level=level['level'])
        # self.print_cls_scores(cls_score, labels, label_weights, num_total_samples)

        if self.use_sigmoid_cls:
            if self.use_focal_loss:
                cls_criterion = weighted_sigmoid_focal_loss
            else:
                cls_criterion = weighted_binary_cross_entropy
        else:
            if self.use_focal_loss:
                raise NotImplementedError
            else:
                cls_criterion = weighted_cross_entropy
        if self.use_focal_loss:
            loss_cls = cls_criterion(
                cls_score,
                labels,
                label_weights,
                gamma=cfg.gamma,
                alpha=cfg.alpha,
                avg_factor=num_total_samples)
        else:
            loss_cls = cls_criterion(
                cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 6)
        bbox_weights = bbox_weights.reshape(-1, 6)
        # 3D images
        bbox_pred = bbox_pred.permute(0, 3, 4, 2, 1).reshape(-1, 6)

        loss_reg = weighted_smoothl1(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=cfg.smoothl1_beta,
            avg_factor=num_total_samples)

        # debug only...
        # print('level {} loss_cls: {}'.format(level['level'], loss_cls))
        # print('level {} loss_reg: {}'.format(level['level'], loss_reg))
        level['level'] += 1
        return loss_cls, loss_reg

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None,
             iteration=None):
        # 3D images
        featmap_sizes = [featmap.size()[-3:] for featmap in cls_scores]

        assert len(featmap_sizes) == len(self.anchor_generators)
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas)
        sampling = False if self.use_focal_loss else True
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = anchor_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=sampling,
            iteration=iteration)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, anchors_list, inside_flags) = cls_reg_targets
        self.pos_indices = inside_flags
        num_total_samples = (num_total_pos if self.use_focal_loss else
                             num_total_pos + num_total_neg)
        level = {'level': 0}
        losses_cls, losses_reg = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            anchors_list,
            num_total_samples=num_total_samples,
            cfg=cfg,
            level=level,
            gt_bboxes=gt_bboxes,
            iteration=iteration)
        return dict(loss_cls=losses_cls, loss_reg=losses_reg)

    def get_bboxes(self, cls_scores, bbox_preds, img_metas, cfg,
                   rescale=False, img_meta_2=None, img_meta_3=None):
        if img_meta_2 is not None:
            img_metas = img_meta_2
        if img_meta_3 is not None:
            img_metas = img_meta_3

        if self.pos_indices_test is None and hasattr(cfg, 'different_img_size') and cfg.different_img_size:
            featmap_sizes = [featmap.size()[-3:] for featmap in cls_scores]
            anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas)
            inside_flags = get_anchor_inside_flags(anchor_list, valid_flag_list, img_metas)
            self.pos_indices_test = inside_flags

        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        mlvl_anchors = [
            self.anchor_generators[i].grid_anchors(cls_scores[i].size()[-3:],
                                                   self.anchor_strides[i], self.anchor_strides_depth[i])
            for i in range(num_levels)
        ]
        result_list, anchors_list = [], []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals, anchors = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               mlvl_anchors, img_shape,
                                               scale_factor, cfg, rescale)
            result_list.append(proposals)
            anchors_list.append(anchors)
        return result_list, anchors_list

    '''
    for debugging 3D images only...
    '''
    def visualize_anchor_boxes(self, imgs, cls_scores, img_metas, slice_num=45, top_k=None, shuffle=False):
        featmap_sizes = [featmap.size()[-3:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas)
        img = tensor2img3D(imgs, slice_num=slice_num)
        anchors = []
        unique_set = set()
        for bboxes in anchor_list[0]:
            bboxes = bboxes.cpu().numpy()
            for bbox in bboxes:
                # select each aspect ratio bounding box in the middle of an image
                # if bbox[0] >= 100 and bbox[0] <= 400 and bbox[2] >= 100 and bbox[2] <= 400 and \
                #     bbox[1] >= 150 and bbox[1] <= 450 and bbox[3] >= 150 and bbox[3] <= 450 and \
                #     slice_num >= bbox[4] and slice_num <= bbox[5] and (bbox[5] - bbox[4]) not in unique_set:
                #     unique_set.add(bbox[5] - bbox[4])
                #     anchors.append([bbox[0], bbox[1], bbox[2], bbox[3]])

                # Get all anchors in the middle of the image
                if bbox[0] >= 100 and bbox[0] <= 400 and bbox[2] >= 100 and bbox[2] <= 400 and \
                    bbox[1] >= 150 and bbox[1] <= 450 and bbox[3] >= 150 and bbox[3] <= 450 and \
                    slice_num >= bbox[4] and slice_num <= bbox[5] and (bbox[2] - bbox[0]) not in unique_set:
                    anchors.append([bbox[0], bbox[1], bbox[2], bbox[3]])
        print(unique_set)
        breakpoint()
        if shuffle is True:
            while True:
                random.shuffle(anchors)
                mmcv.imshow_bboxes(img, np.array(anchors), top_k=20)
        elif top_k is None:
            mmcv.imshow_bboxes(img, np.array(anchors))
        else:
            mmcv.imshow_bboxes(img, np.array(anchors), top_k=top_k)
        breakpoint()

    def imwrite(self, img, file_path, params=None):
        """Write image to file
        """
        return cv2.imwrite(file_path, img, params)

    def visualize_anchors_across_levels(self,
                    bboxes,
                    gt_bboxes=None,
                    labels=None,
                    label_weights=None,
                    colors=[(0, 255, 0), (0, 0, 255), (255, 0, 0)],
                    top_k=-1,
                    thickness=1,
                    show=True,
                    iteration=None,
                    level=None):
        """Draw bboxes on an image.
        """
        if labels.dim() == 3:
            # pick the first patient for display
            cur_labels = labels[0, :, :]
            cur_label_weights = label_weights[0, :, :]
        else:
            cur_labels = labels
            cur_label_weights = label_weights

        pos_indices_labels = [index for index, val in enumerate(cur_labels.cpu().numpy()) if val > 0]
        labels_weights_indices = [index for index, val in enumerate(cur_label_weights.cpu().numpy()) if val > 0]


        # uncomment to print negative boxes
        neg_labels_indices = []
        for label_weight_index in labels_weights_indices:
            if label_weight_index not in pos_indices_labels:
                neg_labels_indices.append(label_weight_index)


        if bboxes.dim() == 3:
            pos_bboxes = bboxes[0, pos_indices_labels, :]
            neg_bboxes = bboxes[0, neg_labels_indices, :]
        else:
            pos_bboxes = bboxes[pos_indices_labels, :]
            neg_bboxes = bboxes[neg_labels_indices, :]

        pos_bboxes = pos_bboxes.cpu().numpy()
        pos_bboxes = [pos_bboxes]
        neg_bboxes = neg_bboxes.cpu().numpy()
        neg_bboxes = [neg_bboxes]
        gt_bboxes = gt_bboxes[0].cpu().numpy()
        gt_bboxes = [gt_bboxes]
        # keys = {'xy', 'xz', 'yz'} 
        keys = {'xy'} 

        for key in keys:
            if key == 'xy':
                img = np.zeros((768, 768, 3))
            elif key == 'xz':
                img = np.zeros((160, 768, 3))
            elif key == 'yz':
                img = np.zeros((160, 768, 3))

            if len(pos_bboxes) > 0: 
                for i, _bboxes in enumerate(pos_bboxes):
                    _bboxes = _bboxes.astype(np.int32)
                    if top_k <= 0:
                        _top_k = _bboxes.shape[0]
                    else:
                        _top_k = min(top_k, _bboxes.shape[0])
                    for j in range(_top_k):
                        if key == 'xy':
                            left_top = (_bboxes[j, 0], _bboxes[j, 1])
                            right_bottom = (_bboxes[j, 2], _bboxes[j, 3])
                        elif key == 'xz':
                            left_top = (_bboxes[j, 0], _bboxes[j, 4])
                            right_bottom = (_bboxes[j, 2], _bboxes[j, 5])
                        elif key == 'yz':
                            left_top = (_bboxes[j, 1], _bboxes[j, 4])
                            right_bottom = (_bboxes[j, 3], _bboxes[j, 5])

                        cv2.rectangle(img, left_top, right_bottom, colors[0], thickness=thickness)
            
            if len(neg_bboxes) > 0: 
                for i, _bboxes in enumerate(neg_bboxes):
                    _bboxes = _bboxes.astype(np.int32)
                    if top_k <= 0:
                        _top_k = _bboxes.shape[0]
                    else:
                        _top_k = min(top_k, _bboxes.shape[0])
                    for j in range(_top_k):
                        if key == 'xy':
                            left_top = (_bboxes[j, 0], _bboxes[j, 1])
                            right_bottom = (_bboxes[j, 2], _bboxes[j, 3])
                        elif key == 'xz':
                            left_top = (_bboxes[j, 0], _bboxes[j, 4])
                            right_bottom = (_bboxes[j, 2], _bboxes[j, 5])
                        elif key == 'yz':
                            left_top = (_bboxes[j, 1], _bboxes[j, 4])
                            right_bottom = (_bboxes[j, 3], _bboxes[j, 5])

                        cv2.rectangle(img, left_top, right_bottom, colors[2], thickness=thickness)
            
            if gt_bboxes is not None and len(gt_bboxes) > 0:
                # gt_bboxes
                for i, _gt_bboxes in enumerate(gt_bboxes):
                    _gt_bboxes = _gt_bboxes.astype(np.int32)
                    if top_k <= 0:
                        _top_k = _gt_bboxes.shape[0]
                    else:
                        _top_k = min(top_k, _gt_bboxes.shape[0])
                    for j in range(_top_k):
                        if key == 'xy':
                            left_top = (_gt_bboxes[j, 0], _gt_bboxes[j, 1])
                            right_bottom = (_gt_bboxes[j, 2], _gt_bboxes[j, 3])
                        elif key == 'xz':
                            left_top = (_gt_bboxes[j, 0], _gt_bboxes[j, 4])
                            right_bottom = (_gt_bboxes[j, 2], _gt_bboxes[j, 5])
                        elif key == 'yz':
                            left_top = (_gt_bboxes[j, 1], _gt_bboxes[j, 4])
                            right_bottom = (_gt_bboxes[j, 3], _gt_bboxes[j, 5])
                        cv2.rectangle( img, left_top, right_bottom, colors[1], thickness=1)
            self.imwrite(img, 'tests/iter_{}_lvl{}_{}.png'.format(iteration, level, key))

    def visualize_anchors_per_slice(self,
                    bboxes,
                    gt_bboxes=None,
                    bbox_weights=None,
                    colors=[(0, 255, 0), (0, 0, 255)],
                    top_k=-1,
                    thickness=1,
                    show=True,
                    iteration=None):
        """Draw bboxes on an image.
        """
        if bbox_weights.dim() == 3:
            # pick the first patient for display
            cur_bbox_weights = bbox_weights[0, :, :]
        else:
            cur_bbox_weights = bbox_weights

        pos_indices_bbox_weights = [index for index, bbox in enumerate(cur_bbox_weights.cpu().numpy()) if bbox[0] > 0]

        if bboxes.dim() == 3:
            bboxes = bboxes[0, pos_indices_bbox_weights, :]
        else:
            bboxes = bboxes[pos_indices_bbox_weights, :]

        bboxes = bboxes.cpu().numpy()
        bboxes = [bboxes]
        gt_bboxes = gt_bboxes[0].cpu().numpy()
        gt_bboxes = [gt_bboxes]
        each_slice_gt = [[] for i in range(160)]
        each_slice_pred = [[] for i in range(160)]

        if gt_bboxes is not None and len(gt_bboxes) > 0:
            # gt_bboxes
            for i, _gt_bboxes in enumerate(gt_bboxes):
                _gt_bboxes = _gt_bboxes.astype(np.int32)
                if top_k <= 0:
                    _top_k = _gt_bboxes.shape[0]
                else:
                    _top_k = min(top_k, _gt_bboxes.shape[0])
                for j in range(_top_k):
                    for cur_slice_num in range(_gt_bboxes[j, 4], _gt_bboxes[j, 5]+1):
                        # left top and right bottom
                        each_slice_gt[cur_slice_num].append(((_gt_bboxes[j, 0], _gt_bboxes[j, 1]), (_gt_bboxes[j, 2], _gt_bboxes[j, 3])))

        if len(bboxes) > 0: 
            # anchors
            for i, _bboxes in enumerate(bboxes):
                _bboxes = _bboxes.astype(np.int32)
                if top_k <= 0:
                    _top_k = _bboxes.shape[0]
                else:
                    _top_k = min(top_k, _bboxes.shape[0])
                for j in range(_top_k):
                    for cur_slice_num in range(_bboxes[j, 4], _bboxes[j, 5]+1):
                        # left top and right bottom
                        each_slice_pred[cur_slice_num].append(((_bboxes[j, 0], _bboxes[j, 1]), (_bboxes[j, 2], _bboxes[j, 3])))
        slice_num = 0
        for slice_gt, slice_pred in zip(each_slice_gt, each_slice_pred):
            # we only print when gt_bbox exists in a slice
            if len(slice_gt) > 0:
                img = np.zeros((768, 768, 3))
                for coords in slice_gt:
                    # red
                    cv2.rectangle( img, coords[0], coords[1], colors[1], thickness=1)
                for coords in slice_pred:
                    # green
                    cv2.rectangle(img, coords[0], coords[1], colors[0], thickness=1)
                self.imwrite(img, 'tests2/iter_{}_slice_{}.png'.format(iteration, slice_num))
            slice_num += 1

    def _expand_binary_labels(self, labels, label_weights, label_channels):
        bin_labels = labels.new_full((labels.size(0), label_channels), 0)
        inds = torch.nonzero(labels >= 1).squeeze()
        if inds.numel() > 0:
            bin_labels[inds, labels[inds] - 1] = 1
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)
        return bin_labels, bin_label_weights

    def print_cls_scores(self, cls_score, labels, label_weights, num_total_samples):
        pos_labels_indices = [index for index, val in enumerate(labels.cpu().numpy()) if val > 0]
        labels_weights_indices = [index for index, val in enumerate(label_weights.cpu().numpy()) if val > 0]

        # make sure every positive label index exist in positive label weights
        for pos_label_index in pos_labels_indices:
            if pos_label_index not in labels_weights_indices:
                breakpoint()
        
        neg_labels_indices = []
        for label_weight_index in labels_weights_indices:
            if label_weight_index not in pos_labels_indices:
                neg_labels_indices.append(label_weight_index)

        label, weight = self._expand_binary_labels(labels, label_weights, cls_score.size(-1))
        losses = F.binary_cross_entropy_with_logits(cls_score, label.float(), weight.float(), reduction='none')[None] / num_total_samples
        adjusted_scores = cls_score.sigmoid()
        pos_scores = adjusted_scores[pos_labels_indices]
        neg_scores = adjusted_scores[neg_labels_indices]
        out = open('output.json', 'a+')

        out.write("number of positive samples: {}\n".format(len(pos_labels_indices)))
        out.write("number of negative samples: {}\n".format(len(neg_labels_indices)))

        cur_counter = 0
        max_to_print = 20 # only print this number of entries
        for index, score, loss in zip(pos_labels_indices, pos_scores, losses.flatten()[pos_labels_indices]):
            if cur_counter == max_to_print:
                break
            out.write("positive sample's index {}, score {}, loss {}\n".format(index, score.data, loss))
            cur_counter += 1
        out.write("\n")

        cur_counter = 0
        for index, score, loss in zip(neg_labels_indices, neg_scores, losses.flatten()[neg_labels_indices]):
            if cur_counter == max_to_print:
                break
            out.write("negative sample's index {}, score {}, loss {}\n".format(index, score.data, loss))
            cur_counter += 1