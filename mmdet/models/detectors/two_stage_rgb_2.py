import torch
import torch.nn as nn

from .base import BaseDetector
from .test_mixins_rgb import RPNTestMixin, BBoxTestMixin, MaskTestMixin
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler


@DETECTORS.register_module
class TwoStageDetectorRGB2(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStageDetectorRGB2, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if shared_head is not None:
            self.shared_heads = nn.ModuleList([builder.build_shared_head(shared_head), builder.build_shared_head(shared_head), builder.build_shared_head(shared_head)])

        if rpn_head is not None:
            self.rpn_heads = nn.ModuleList([builder.build_head(rpn_head), builder.build_head(rpn_head), builder.build_head(rpn_head)])

        if bbox_head is not None:
            self.bbox_roi_extractors = nn.ModuleList([builder.build_roi_extractor(bbox_roi_extractor), builder.build_roi_extractor(bbox_roi_extractor), builder.build_roi_extractor(bbox_roi_extractor)])
            self.bbox_heads = nn.ModuleList([builder.build_head(bbox_head), builder.build_head(bbox_head), builder.build_head(bbox_head)])
            
        if mask_head is not None:
            if mask_roi_extractor is not None:
                self.mask_roi_extractors = nn.ModuleList([builder.build_roi_extractor(mask_roi_extractor), builder.build_roi_extractor(mask_roi_extractor), builder.build_roi_extractor(mask_roi_extractor)])
                self.share_roi_extractor = False
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractors = self.bbox_roi_extractors
            self.mask_heads = nn.ModuleList([builder.build_head(mask_head), builder.build_head(mask_head), builder.build_head(mask_head)])

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_heads') and self.rpn_heads is not None

    @property
    def with_shared_head(self):
        return hasattr(self, 'shared_heads') and self.shared_heads is not None

    @property
    def with_bbox(self):
        return hasattr(self, 'bbox_heads') and self.bbox_heads is not None

    @property
    def with_mask(self):
        return hasattr(self, 'mask_heads') and self.mask_heads is not None

    def init_weights(self, pretrained=None):
        super(TwoStageDetectorRGB2, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_shared_head:
            for i in range(len(self.shared_heads)):
                self.shared_heads[i].init_weights(pretrained=pretrained)
        if self.with_rpn:
            for i in range(len(self.rpn_heads)):
                self.rpn_heads[i].init_weights()
        if self.with_bbox:
            for i in range(len(self.bbox_roi_extractors)):
                self.bbox_roi_extractors[i].init_weights()
            for i in range(len(self.bbox_heads)):
                self.bbox_heads[i].init_weights()
        if self.with_mask:
            for i in range(len(self.mask_heads)):
                self.mask_heads[i].init_weights()
            if not self.share_roi_extractor:
                 for i in range(len(self.mask_roi_extractors)):
                    self.mask_roi_extractors[i].init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def append_to_losses(self, losses, cur_loss):
        keys = cur_loss.keys()
        for key in keys:
            value = cur_loss[key]

            if key in losses:
                if isinstance(value, list):
                    for each_val in value:
                        losses[key].append(each_val)
                else:
                    losses[key].append(value)
            else:
                if isinstance(value, list):
                    losses[key] = value
                else:
                    # not a list, turn into a list
                    losses[key] = [value]

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        x = self.extract_feat(img)

        losses= dict()

        num_slices = 3
        
        for index in range(num_slices):
            cur_bboxes = gt_bboxes[index]
            cur_labels = gt_labels[index]
            cur_bboxes_ignore = None if gt_bboxes_ignore is None else gt_bboxes_ignore[index]
            cur_masks = None if gt_masks is None else gt_masks[index]
            cur_proposals = None if proposals is None else proposals[index]

            # RPN forward and loss
            if self.with_rpn:
                rpn_outs = self.rpn_heads[index](x)
                rpn_loss_inputs = rpn_outs + (cur_bboxes, img_meta,
                                            self.train_cfg.rpn)
                rpn_losses = self.rpn_heads[index].loss(
                    *rpn_loss_inputs, gt_bboxes_ignore=cur_bboxes_ignore)
                self.append_to_losses(losses, rpn_losses)

                proposal_inputs = rpn_outs + (img_meta, self.test_cfg.rpn)
                proposal_list = self.rpn_heads[index].get_bboxes(*proposal_inputs)
            else:
                proposal_list = cur_proposals

            # assign gts and sample proposals
            if self.with_bbox or self.with_mask:
                bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
                bbox_sampler = build_sampler(
                    self.train_cfg.rcnn.sampler, context=self)
                num_imgs = img.size(0)
                if cur_bboxes_ignore is None or len(cur_bboxes_ignore) == 0:
                    cur_bboxes_ignore = [None for _ in range(num_imgs)]
                sampling_results = []
                for i in range(num_imgs):
                    assign_result = bbox_assigner.assign(
                        proposal_list[i], cur_bboxes[i], cur_bboxes_ignore[i],
                        cur_labels[i])
                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        proposal_list[i],
                        cur_bboxes[i],
                        cur_labels[i],
                        feats=[lvl_feat[i][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)

            # bbox head forward and loss
            if self.with_bbox:
                rois = bbox2roi([res.bboxes for res in sampling_results])
                # TODO: a more flexible way to decide which feature maps to use
                bbox_feats = self.bbox_roi_extractors[index](
                    x[:self.bbox_roi_extractors[index].num_inputs], rois)
                if self.with_shared_head:
                    bbox_feats = self.shared_heads[index](bbox_feats)
                cls_score, bbox_pred = self.bbox_heads[index](bbox_feats)

                bbox_targets = self.bbox_heads[index].get_target(
                    sampling_results, cur_bboxes, cur_labels, self.train_cfg.rcnn)
                loss_bbox = self.bbox_heads[index].loss(cls_score, bbox_pred,
                                                *bbox_targets)
                self.append_to_losses(losses, loss_bbox)

            # mask head forward and loss
            if self.with_mask:
                if not self.share_roi_extractor:
                    pos_rois = bbox2roi(
                        [res.pos_bboxes for res in sampling_results])
                    mask_feats = self.mask_roi_extractors[index](
                        x[:self.mask_roi_extractors[index].num_inputs], pos_rois)
                    if self.with_shared_head:
                        mask_feats = self.shared_heads[index](mask_feats)
                else:
                    pos_inds = []
                    device = bbox_feats.device
                    for res in sampling_results:
                        pos_inds.append(
                            torch.ones(
                                res.pos_bboxes.shape[0],
                                device=device,
                                dtype=torch.uint8))
                        pos_inds.append(
                            torch.zeros(
                                res.neg_bboxes.shape[0],
                                device=device,
                                dtype=torch.uint8))
                    pos_inds = torch.cat(pos_inds)
                    mask_feats = bbox_feats[pos_inds]
                mask_pred = self.mask_heads[index](mask_feats)

                mask_targets = self.mask_heads[index].get_target(
                    sampling_results, cur_masks, self.train_cfg.rcnn)
                pos_labels = torch.cat(
                    [res.pos_gt_labels for res in sampling_results])
                loss_mask = self.mask_heads[index].loss(mask_pred, mask_targets,
                                                pos_labels)
                self.append_to_losses(losses, loss_mask)

        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x = self.extract_feat(img)

        num_slices = 3
        bbox_results_list, segm_results_list = [], []
        for index in range(num_slices):
            proposal_list = self.simple_test_rpn(
                x, img_meta, self.test_cfg.rpn, slice_num=index) if proposals is None else proposals

            det_bboxes, det_labels = self.simple_test_bboxes(
                x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale, slice_num=index)
            bbox_results = bbox2result(det_bboxes, det_labels,
                                    self.bbox_heads[index].num_classes)

            bbox_results_list.append(bbox_results)
            if self.with_mask:
                segm_results = self.simple_test_mask(
                    x, img_meta, det_bboxes, det_labels, rescale=rescale, slice_num=index)
                segm_results_list.append(segm_results)

        if not self.with_mask:
            return bbox_results_list
        else:
            return bbox_results_list, segm_results_list


    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        breakpoint()
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,
            self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results
