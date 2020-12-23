import torch
import torch.nn as nn

from .base import BaseDetector
from .test_mixins import RPNTestMixin, BBoxTestMixin, MaskTestMixin
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler, tensor2imgs
import numpy as np
import mmcv
import matplotlib.pyplot as plt
import cv2


@DETECTORS.register_module
class TwoStageDetector(BaseDetector, RPNTestMixin, BBoxTestMixin,
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
        super(TwoStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.iteration = 1
        self.iterations = []
        self.rpn_cls_losses = []
        self.rpn_bbox_reg_losses = []
        self.total_losses = []

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)

        # self.mask_head = None; mask_head = None

        if mask_head is not None:
            if mask_roi_extractor is not None:
                self.mask_roi_extractor = builder.build_roi_extractor(
                    mask_roi_extractor)
                self.share_roi_extractor = False
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.bbox_roi_extractor
            self.mask_head = builder.build_head(mask_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(TwoStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_inputs = rpn_outs + (img_meta, self.test_cfg.rpn)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
            # self.visualize_proposals(img, proposal_list, gt_bboxes) #debug only
            # self.visualize_gt_bboxes(img, gt_bboxes, img_meta) # debug only
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)

        # mask head forward and loss
        if self.with_mask:
            if not self.share_roi_extractor:
                pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])
                mask_feats = self.mask_roi_extractor(
                    x[:self.mask_roi_extractor.num_inputs], pos_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
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
            mask_pred = self.mask_head(mask_feats)

            mask_targets = self.mask_head.get_target(
                sampling_results, gt_masks, self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)
            losses.update(loss_mask)

        # self.save_losses_plt(losses) #debug only...
        self.iteration += 1
        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x = self.extract_feat(img)

        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
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

    def visualize_proposals(self, imgs, proposals, gt_bboxes):
        img = tensor2imgs(imgs, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))

        batch_num = 0
        for cur_proposals, cur_gt_bboxes in zip(proposals, gt_bboxes):
            bboxes = []
            cur_proposals = cur_proposals.cpu().numpy()
            for bbox in cur_proposals:
                bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
            filename = 'tests2/iter_{}_batch_{}.png'.format(self.iteration, batch_num)
            self.show_bboxes_gt_bboxes(img[0], np.array(bboxes), gt_bboxes=cur_gt_bboxes, out_file=filename)
            batch_num += 1
    
    def save_losses_plt(self, losses):
        rpn_cls_loss = torch.sum(torch.stack(losses['loss_rpn_cls'])).item()
        rpn_bbox_reg_loss = torch.sum(torch.stack(losses['loss_rpn_reg'])).item()
        self.rpn_cls_losses.append(rpn_cls_loss)
        self.rpn_bbox_reg_losses.append(rpn_bbox_reg_loss)
        self.total_losses.append(rpn_cls_loss + rpn_bbox_reg_loss)
        self.iterations.append(self.iteration)
        plt.plot(self.iterations, self.rpn_cls_losses, color='skyblue', linewidth=1, label='class loss')
        plt.plot(self.iterations, self.rpn_bbox_reg_losses, color='olive', linewidth=1, label='bbox reg loss')
        plt.plot(self.iterations, self.total_losses, color='red', linewidth=1, label='total loss')
        if self.iteration == 1:
            plt.legend()
        plt.savefig('tests2/iter_{}_loss.png'.format(self.iteration))

    def imwrite(self, img, file_path, params=None):
        """Write image to file
        """
        return cv2.imwrite(file_path, img, params)

    def visualize_gt_bboxes(self, imgs, gt_bboxes, img_meta):
        imgs = tensor2imgs(imgs, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))
        batch_num = 1
        for img, gt_bbox in zip(imgs, gt_bboxes):
            filename = 'tests/iter_{}_batchnum_{}.png'.format(self.iteration, batch_num)
            mmcv.imshow_bboxes(img, gt_bbox.cpu().numpy(), show=False, out_file=filename)
            batch_num += 1
        breakpoint()

    def show_bboxes_gt_bboxes(self, img,
                    bboxes,
                    gt_bboxes=None,
                    colors=[(0, 255, 0), (0, 0, 255)],
                    top_k=-1,
                    thickness=1,
                    show=True,
                    out_file=None):
        """Draw bboxes on an image.
        """
        if isinstance(bboxes, np.ndarray):
            bboxes = [bboxes]

        for i, _bboxes in enumerate(bboxes):
            _bboxes = _bboxes.astype(np.int32)
            if top_k <= 0:
                _top_k = _bboxes.shape[0]
            else:
                _top_k = min(top_k, _bboxes.shape[0])
            for j in range(_top_k):
                left_top = (_bboxes[j, 0], _bboxes[j, 1])
                right_bottom = (_bboxes[j, 2], _bboxes[j, 3])
                cv2.rectangle(img, left_top, right_bottom, colors[0], thickness=thickness)
        
        if gt_bboxes is not None:
            gt_bboxes = gt_bboxes.cpu().numpy()
            if isinstance(gt_bboxes, np.ndarray):
                gt_bboxes = [gt_bboxes]

            # gt_bboxes
            for i, gt_bboxes in enumerate(gt_bboxes):
                _gt_bboxes = gt_bboxes.astype(np.int32)
                if top_k <= 0:
                    _top_k = _gt_bboxes.shape[0]
                else:
                    _top_k = min(top_k, _gt_bboxes.shape[0])
                for j in range(_top_k):
                    left_top = (_gt_bboxes[j, 0], _gt_bboxes[j, 1])
                    right_bottom = (_gt_bboxes[j, 2], _gt_bboxes[j, 3])
                    cv2.rectangle( img, left_top, right_bottom, colors[1], thickness=1)

        self.imwrite(img, out_file)