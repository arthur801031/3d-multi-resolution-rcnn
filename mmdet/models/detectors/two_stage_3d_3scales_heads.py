import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseDetector
from .test_mixins_3d import RPNTestMixin, BBoxTestMixin, MaskTestMixin
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2roi3D, bbox2result3D, build_assigner, build_sampler, tensor2img3D, multiclass_nms_3d
import mmcv
import math
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as pts
import numpy as np


@DETECTORS.register_module
class TwoStageDetector3D3ScalesHeads(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 rpn_head=None,
                 rpn_head_2=None,
                 rpn_head_3=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 refinement_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStageDetector3D3ScalesHeads, self).__init__()

        # for debugging....
        self.iteration = 1
        self.iterations = []
        self.rpn_cls_losses = []
        self.rpn_bbox_reg_losses = []
        self.total_losses = []

        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)
        if rpn_head_2 is not None:
            self.rpn_head_2 = builder.build_head(rpn_head_2)
        if rpn_head_3 is not None:
            self.rpn_head_3 = builder.build_head(rpn_head_3)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_roi_extractor_2 = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_roi_extractor_3 = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)
            self.bbox_head_2 = builder.build_head(bbox_head)
            self.bbox_head_3 = builder.build_head(bbox_head)

            self.bbox_roi_extractor_refinement_2 = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_roi_extractor_refinement_3 = builder.build_roi_extractor(
                bbox_roi_extractor)
        if refinement_head is not None:
            self.refinement_head_2 = builder.build_head(refinement_head)
            self.refinement_head_3 = builder.build_head(refinement_head)
        else:
            self.refinement_head_2 = None
            self.refinement_head_3 = None

        self.mask_head = None; mask_head = None

        if mask_head is not None:
            if mask_roi_extractor is not None:
                self.mask_roi_extractor = builder.build_roi_extractor(
                    mask_roi_extractor)
                self.mask_roi_extractor_refinement_2 = builder.build_roi_extractor(
                    mask_roi_extractor)
                self.mask_roi_extractor_refinement_3 = builder.build_roi_extractor(
                    mask_roi_extractor)
                self.share_roi_extractor = False
            else:
                breakpoint()
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.bbox_roi_extractor
            self.mask_head = builder.build_head(mask_head)
            # enable second mask head
            self.mask_head_refinement_2 = builder.build_head(mask_head)
            # enable third mask head
            self.mask_head_refinement_3 = builder.build_head(mask_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(TwoStageDetector3D3ScalesHeads, self).init_weights(pretrained)
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
            self.rpn_head_2.init_weights()
            self.rpn_head_3.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_roi_extractor_2.init_weights()
            self.bbox_roi_extractor_3.init_weights()
            self.bbox_roi_extractor_refinement_2.init_weights()
            self.bbox_roi_extractor_refinement_3.init_weights()
            self.bbox_head.init_weights()
            self.bbox_head_2.init_weights()
            self.bbox_head_3.init_weights()
        if self.refinement_head_2 and self.refinement_head_3:
            self.refinement_head_2.init_weights()
            self.refinement_head_3.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            # enable second mask head
            self.mask_head_refinement_2.init_weights()
            # enable third mask head
            self.mask_head_refinement_3.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()
                # enable second mask head
                self.mask_roi_extractor_refinement_2.init_weights()
                # enable third mask head
                self.mask_roi_extractor_refinement_3.init_weights()

    def extract_feat(self, imgs):
        x = self.backbone(imgs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, imgs, img_meta, imgs_2, img_meta_2, imgs_3, img_meta_3, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(imgs, img_meta, imgs_2, img_meta_2, imgs_3, img_meta_3, **kwargs)
        else:
            return self.forward_test(imgs, img_meta,  imgs_2, img_meta_2, imgs_3, img_meta_3, **kwargs)

    def forward_train(self,
                      imgs,
                      img_meta,
                      imgs_2,
                      img_meta_2,
                      imgs_3,
                      img_meta_3,
                      gt_bboxes,
                      gt_bboxes_2,
                      gt_bboxes_3,
                      gt_labels,
                      gt_labels_2,
                      gt_labels_3,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_masks_2=None,
                      gt_masks_3=None,
                      pp=None,
                      pp_2=None,
                      proposals=None):
        # self.print_iterations()
        assert imgs.shape[1] == 3 and imgs_2.shape[1] == 3 and imgs_3.shape[1] == 3 # make sure channel size is 3

        # Default FPN
        x = self.extract_feat(imgs)
        x_2 = self.extract_feat(imgs_2)
        x_3 = self.extract_feat(imgs_3)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_outs_2 = self.rpn_head_2(x_2)
            rpn_outs_3 = self.rpn_head_3(x_3)

            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_loss_inputs_2 = rpn_outs_2 + (gt_bboxes_2, img_meta,
                                          self.train_cfg.rpn)
            rpn_loss_inputs_3 = rpn_outs_3 + (gt_bboxes_3, img_meta,
                                          self.train_cfg.rpn)

            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, iteration=self.iteration)
            rpn_losses_2 = self.rpn_head_2.loss(
                *rpn_loss_inputs_2, gt_bboxes_ignore=gt_bboxes_ignore, iteration=self.iteration, img_meta_2=img_meta_2)
            rpn_losses_3 = self.rpn_head_3.loss(
                *rpn_loss_inputs_3, gt_bboxes_ignore=gt_bboxes_ignore, iteration=self.iteration, img_meta_3=img_meta_3)

            losses.update(rpn_losses)
            losses.update(rpn_losses_2)
            losses.update(rpn_losses_3)

            proposal_inputs = rpn_outs + (img_meta, self.train_cfg.rpn_proposal)
            proposal_inputs_2 = rpn_outs_2 + (img_meta, self.train_cfg.rpn_proposal)
            proposal_inputs_3 = rpn_outs_3 + (img_meta, self.train_cfg.rpn_proposal)

            proposal_list, anchors = self.rpn_head.get_bboxes(*proposal_inputs)
            proposal_list_2, anchors_2 = self.rpn_head_2.get_bboxes(*proposal_inputs_2, img_meta_2=img_meta_2)
            proposal_list_3, anchors_3 = self.rpn_head_3.get_bboxes(*proposal_inputs_3, img_meta_3=img_meta_3)


            # self.rpn_head.visualize_anchor_boxes(imgs, rpn_outs[0], img_meta, slice_num=45, shuffle=True) # debug only
            # self.visualize_proposals(imgs, proposal_list, gt_bboxes, img_meta, slice_num=None, isProposal=True) #debug only
            # self.visualize_proposals(imgs, anchors, gt_bboxes, img_meta, slice_num=None, isProposal=False) #debug only
            # self.visualize_gt_bboxes(imgs, gt_bboxes, img_meta) #debug only
            # breakpoint()
            # self.visualize_gt_bboxes(imgs_2, gt_bboxes_2, img_meta_2) #debug only
            # breakpoint()
            # self.visualize_gt_bboxes(imgs_3, gt_bboxes_3, img_meta_3) #debug only
            # breakpoint()
            # self.visualize_gt_bboxes_masks(imgs_2, gt_bboxes_2, img_meta_2, gt_masks) # debug only
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = imgs.size(0)
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []

            for i in range(num_imgs):
                gt_bboxes_cur_pat = gt_bboxes[i]
                gt_bboxes_ignore_cur_pat = gt_bboxes_ignore[i]
                gt_labels_cur_pat = gt_labels[i]

                assign_result = bbox_assigner.assign(
                    proposal_list[i], gt_bboxes_cur_pat, 
                    gt_bboxes_ignore_cur_pat, gt_labels_cur_pat)
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes_cur_pat,
                    gt_labels_cur_pat,
                    feats=[lvl_feat[i][None] for lvl_feat in x])     
                sampling_results.append(sampling_result)

            bbox_assigner_2 = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler_2 = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs_2 = imgs_2.size(0)
            gt_bboxes_ignore_2 = [None for _ in range(num_imgs_2)]
            sampling_results_2 = []

            for i in range(num_imgs_2):
                gt_bboxes_cur_pat_2 = gt_bboxes_2[i]
                gt_bboxes_ignore_cur_pat_2 = gt_bboxes_ignore_2[i]
                gt_labels_cur_pat_2 = gt_labels_2[i]

                assign_result_2 = bbox_assigner_2.assign(
                    proposal_list_2[i], gt_bboxes_cur_pat_2, 
                    gt_bboxes_ignore_cur_pat_2, gt_labels_cur_pat_2)
                sampling_result_2 = bbox_sampler_2.sample(
                    assign_result_2,
                    proposal_list_2[i],
                    gt_bboxes_cur_pat_2,
                    gt_labels_cur_pat_2,
                    feats=[lvl_feat[i][None] for lvl_feat in x_2])     
                sampling_results_2.append(sampling_result_2)
            
            bbox_assigner_3 = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler_3 = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs_3 = imgs_3.size(0)
            gt_bboxes_ignore_3 = [None for _ in range(num_imgs_3)]
            sampling_results_3 = []

            for i in range(num_imgs_3):
                gt_bboxes_cur_pat_3 = gt_bboxes_3[i]
                gt_bboxes_ignore_cur_pat_3 = gt_bboxes_ignore_3[i]
                gt_labels_cur_pat_3 = gt_labels_3[i]

                assign_result_3 = bbox_assigner_3.assign(
                    proposal_list_3[i], gt_bboxes_cur_pat_3, 
                    gt_bboxes_ignore_cur_pat_3, gt_labels_cur_pat_3)
                sampling_result_3 = bbox_sampler_3.sample(
                    assign_result_3,
                    proposal_list_3[i],
                    gt_bboxes_cur_pat_3,
                    gt_labels_cur_pat_3,
                    feats=[lvl_feat[i][None] for lvl_feat in x_3])     
                sampling_results_3.append(sampling_result_3)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi3D([res.bboxes for res in sampling_results])
            rois_2 = bbox2roi3D([res.bboxes for res in sampling_results_2])
            rois_3 = bbox2roi3D([res.bboxes for res in sampling_results_3])

            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            bbox_feats_2 = self.bbox_roi_extractor_2(
                x_2[:self.bbox_roi_extractor_2.num_inputs], rois_2)
            bbox_feats_3 = self.bbox_roi_extractor_3(
                x_3[:self.bbox_roi_extractor_3.num_inputs], rois_3)

            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            cls_score_2, bbox_pred_2 = self.bbox_head_2(bbox_feats_2)
            cls_score_3, bbox_pred_3 = self.bbox_head_3(bbox_feats_3)

            bbox_targets = self.bbox_head.get_target(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)
            bbox_targets_2 = self.bbox_head_2.get_target(
                sampling_results_2, gt_bboxes_2, gt_labels_2, self.train_cfg.rcnn)
            bbox_targets_3 = self.bbox_head_3.get_target(
                sampling_results_3, gt_bboxes_3, gt_labels_3, self.train_cfg.rcnn)
            
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            loss_bbox_2 = self.bbox_head_2.loss(cls_score_2, bbox_pred_2,
                                            *bbox_targets_2, img_meta_2=img_meta_2)
            loss_bbox_3 = self.bbox_head_3.loss(cls_score_3, bbox_pred_3,
                                            *bbox_targets_3, img_meta_3=img_meta_3)

            losses.update(loss_bbox)
            losses.update(loss_bbox_2)
            losses.update(loss_bbox_3)

        if self.refinement_head_2:
            # prepare upscaled data for refinement head
            upscaled_factor_2 = img_meta_2[0]['ori_shape'][0] / img_meta[0]['ori_shape'][0]
            # convert parameterized adjustments to actual bounding boxes coordinates
            pred_bboxes_2 = self.bbox_head_2.convert_adjustments_to_bboxes(rois_2, bbox_pred_2, img_meta_2[0]['img_shape'])[:,6:].cpu().detach().numpy() / upscaled_factor_2
            
            pred_cls_score_2 = cls_score_2[:,1, None].cpu().detach().numpy()
            pred_bboxes_2 = np.concatenate((pred_bboxes_2, pred_cls_score_2), axis=1)
            pred_bboxes_2 = [torch.from_numpy(pred_bboxes_2).cuda()]
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = imgs.size(0)
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results_refinement_2 = []
            for i in range(num_imgs):
                gt_bboxes_cur_pat = gt_bboxes[i]
                gt_bboxes_ignore_cur_pat = gt_bboxes_ignore[i]
                gt_labels_cur_pat = gt_labels[i]

                assign_result = bbox_assigner.assign(
                    pred_bboxes_2[i], gt_bboxes_cur_pat, 
                    gt_bboxes_ignore_cur_pat, gt_labels_cur_pat)
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    pred_bboxes_2[i],
                    gt_bboxes_cur_pat,
                    gt_labels_cur_pat,
                    feats=[lvl_feat[i][None] for lvl_feat in x])     
                sampling_results_refinement_2.append(sampling_result)
            rois_refinement_2 = bbox2roi3D([res.bboxes for res in sampling_results_refinement_2])
            bbox_feats_refinement_2 = self.bbox_roi_extractor_refinement_2(
                x[:self.bbox_roi_extractor_refinement_2.num_inputs], rois_refinement_2)
            # training refinement head
            refined_bbox_pred_2 = self.refinement_head_2(bbox_feats_refinement_2)
            bbox_targets_refinement_2 = self.refinement_head_2.get_target(
                sampling_results_refinement_2, gt_bboxes, gt_labels, self.train_cfg.rcnn)
            loss_refinement_2 = self.refinement_head_2.loss(refined_bbox_pred_2,
                                            *bbox_targets_refinement_2, img_meta_2=img_meta_2)
            losses.update(loss_refinement_2)

        if self.refinement_head_3:
            # prepare upscaled data for refinement head
            upscaled_factor_3 = img_meta_3[0]['ori_shape'][0] / img_meta[0]['ori_shape'][0]
            # convert parameterized adjustments to actual bounding boxes coordinates
            pred_bboxes_3 = self.bbox_head_3.convert_adjustments_to_bboxes(rois_3, bbox_pred_3, img_meta_3[0]['img_shape'])[:,6:].cpu().detach().numpy() / upscaled_factor_3
            
            pred_cls_score_3 = cls_score_3[:,1, None].cpu().detach().numpy()
            pred_bboxes_3 = np.concatenate((pred_bboxes_3, pred_cls_score_3), axis=1)
            pred_bboxes_3 = [torch.from_numpy(pred_bboxes_3).cuda()]
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = imgs.size(0)
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results_refinement_3 = []
            for i in range(num_imgs):
                gt_bboxes_cur_pat = gt_bboxes[i]
                gt_bboxes_ignore_cur_pat = gt_bboxes_ignore[i]
                gt_labels_cur_pat = gt_labels[i]

                assign_result = bbox_assigner.assign(
                    pred_bboxes_3[i], gt_bboxes_cur_pat, 
                    gt_bboxes_ignore_cur_pat, gt_labels_cur_pat)
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    pred_bboxes_3[i],
                    gt_bboxes_cur_pat,
                    gt_labels_cur_pat,
                    feats=[lvl_feat[i][None] for lvl_feat in x])     
                sampling_results_refinement_3.append(sampling_result)
            rois_refinement_3 = bbox2roi3D([res.bboxes for res in sampling_results_refinement_3])
            bbox_feats_refinement_3 = self.bbox_roi_extractor_refinement_3(
                x[:self.bbox_roi_extractor_refinement_3.num_inputs], rois_refinement_3)
            # training refinement head
            refined_bbox_pred_3 = self.refinement_head_3(bbox_feats_refinement_3)
            bbox_targets_refinement_3 = self.refinement_head_3.get_target(
                sampling_results_refinement_3, gt_bboxes, gt_labels, self.train_cfg.rcnn)
            loss_refinement_3 = self.refinement_head_3.loss(refined_bbox_pred_3,
                                            *bbox_targets_refinement_3, img_meta_3=img_meta_3)
            losses.update(loss_refinement_3)


        # mask head forward and loss
        if self.with_mask:
            # lower resolution mask head
            pos_rois = bbox2roi3D(
                [res.pos_bboxes for res in sampling_results])
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], pos_rois)
            mask_pred = self.mask_head(mask_feats)
            mask_targets = self.mask_head.get_target(
                sampling_results, gt_masks, self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)
            losses.update(loss_mask)

            # refinement head mask head 2
            pos_rois = bbox2roi3D(
                [res.pos_bboxes for res in sampling_results_refinement_2])
            mask_feats = self.mask_roi_extractor_refinement_2(
                x[:self.mask_roi_extractor_refinement_2.num_inputs], pos_rois)
            mask_pred = self.mask_head_refinement_2(mask_feats)
            mask_targets = self.mask_head_refinement_2.get_target(
                sampling_results_refinement_2, gt_masks, self.train_cfg.rcnn)            
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results_refinement_2])
            loss_mask_refinement_2 = self.mask_head_refinement_2.loss(mask_pred, mask_targets,
                                            pos_labels, img_meta_2=img_meta_2)
            losses.update(loss_mask_refinement_2)

            # refinement head mask head 3
            pos_rois = bbox2roi3D(
                [res.pos_bboxes for res in sampling_results_refinement_3])
            mask_feats = self.mask_roi_extractor_refinement_3(
                x[:self.mask_roi_extractor_refinement_3.num_inputs], pos_rois)
            mask_pred = self.mask_head_refinement_3(mask_feats)
            mask_targets = self.mask_head_refinement_3.get_target(
                sampling_results_refinement_3, gt_masks, self.train_cfg.rcnn)            
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results_refinement_3])
            loss_mask_refinement_3 = self.mask_head_refinement_3.loss(mask_pred, mask_targets,
                                            pos_labels, img_meta_3=img_meta_3)
            losses.update(loss_mask_refinement_3)

        # self.save_losses_plt(losses) #debug only...
        self.iteration += 1
        return losses

    def forward_test(self, imgs, img_metas, imgs_2, img_meta_2, imgs_3, img_meta_3, **kwargs):
        return self.simple_test(imgs, img_metas, imgs_2, img_meta_2, imgs_3, img_meta_3, **kwargs)

    def simple_test(self, imgs, img_metas, imgs_2, img_metas_2, imgs_3, img_metas_3, pp=None, pp_2=None, proposals=None, rescale=False, test_cfg2=None, test_cfg3=None):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."
        
        if test_cfg3 is not None:
            test_cfg = test_cfg3
        elif test_cfg2 is not None:
            test_cfg = test_cfg2
        else:
            test_cfg = self.test_cfg

        img_metas = img_metas[0]
        img_metas_2 = img_metas_2[0]
        img_metas_3 = img_metas_3[0]

        # Default FPN
        x = self.extract_feat(imgs)
        x_2 = self.extract_feat(imgs_2)
        x_3 = self.extract_feat(imgs_3)

        # dataset 1
        proposal_list = self.simple_test_rpn(
            x, img_metas, test_cfg.rpn) if proposals is None else proposals
        bboxes, scores = self.simple_test_bboxes(
            x, img_metas, proposal_list, None, rescale=rescale)
        del proposal_list

        # dataset 2
        proposal_list = self.simple_test_rpn_2(
            x_2, img_metas_2, test_cfg.rpn) if proposals is None else proposals
        bboxes_2, scores_2 = self.simple_test_bboxes_2(
            x_2, img_metas_2, proposal_list, None, rescale=rescale)
        del proposal_list
        
        # dataset 3
        proposal_list = self.simple_test_rpn_3(
            x_3, img_metas_3, test_cfg.rpn) if proposals is None else proposals
        bboxes_3, scores_3 = self.simple_test_bboxes_3(
            x_3, img_metas_3, proposal_list, None, rescale=rescale)
        del proposal_list

        # refinement head
        if self.refinement_head_2 and self.refinement_head_3:
            bboxes_2_refinement = bboxes_2[:, 6:]
            bboxes_2_refinement = [torch.cat((bboxes_2_refinement, scores_2[:,1, None]), dim=1)]
            bboxes_2_refinement = self.simple_test_bbox_refinement_2(
                x, img_metas, bboxes_2_refinement, None, rescale=rescale)
            
            bboxes_3_refinement = bboxes_3[:, 6:]
            bboxes_3_refinement = [torch.cat((bboxes_3_refinement, scores_3[:,1, None]), dim=1)]
            bboxes_3_refinement = self.simple_test_bbox_refinement_3(
                x, img_metas, bboxes_3_refinement, None, rescale=rescale)

            bboxes_final = torch.cat((bboxes, bboxes_2_refinement, bboxes_3_refinement), 0)
            scores_final = torch.cat((scores, scores_2, scores_3), 0)
        else:
            bboxes_final = torch.cat((bboxes, bboxes_2, bboxes_3), 0)
            scores_final = torch.cat((scores, scores_2, scores_3), 0)
        det_bboxes, det_labels = multiclass_nms_3d(bboxes_final, scores_final, test_cfg.rcnn.score_thr, test_cfg.rcnn.nms, test_cfg.rcnn.max_per_img)
        
        bbox_results = bbox2result3D(det_bboxes, det_labels,
                                        self.bbox_head.num_classes)

        # free CUDA memory otherwise it would run out of memory
        del x; del x_2; del x_3
        del bboxes; del bboxes_2; del bboxes_3;
        del scores; del scores_2; del scores_3
        del bboxes_final; del scores_final
        del det_bboxes; del det_labels
        if self.refinement_head_2 and self.refinement_head_3:
            del bboxes_2_refinement
            del bboxes_3_refinement
        torch.cuda.empty_cache()

        # ############ test RPN's performance ############
        # proposal_list = proposal_list[0].cpu().numpy()
        # return [proposal_list]

        # ############ only return bbox ############
        return bbox_results


        # if not self.with_mask:
        #     return bbox_results
        # else:
        #     segm_results = self.simple_test_mask(
        #         x, img_metas, det_bboxes, det_labels, rescale=rescale)
        #     # self.visualize_masks_bboxes(bbox_results, segm_results) # debug only
        #     return bbox_results, segm_results

    def print_iterations(self):
        out = open('output.json', 'a+')
        out.write('\n\n\nEpoch/iteration: {}\n'.format(self.iteration))


    def visualize_masks_bboxes(self, bbox_results, segm_results):
        bbox_results = bbox_results[0]
        segm_results = segm_results[0]
        i = 1
        for bbox, segm in zip(bbox_results, segm_results):
            print('bbox: ', bbox)
            exist_slices = []
            for cur_slice in range(segm.shape[0]):
                if len(np.unique(segm[cur_slice,:,:])) > 1:
                    exist_slices.append(cur_slice)
            assert int(bbox[4]) == exist_slices[0] and int(bbox[5]) == exist_slices[-1]

            for cur_slice in exist_slices:
                filename = 'tests/bbox {} slice {}.png'.format(i, cur_slice)
                plt.figure()
                plt.imshow(segm[cur_slice,:,:])
                ax = plt.gca()
                rect = pts.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
                plt.savefig(filename)
                plt.close()
            i += 1
        breakpoint()

    def visualize_proposals(self, imgs, proposals, gt_bboxes, img_meta, slice_num, isProposal=True):
        if slice_num is None:
            img = tensor2img3D(imgs, slice_num=45)
        else:
            img = tensor2img3D(imgs, slice_num=slice_num)

        batch_num = 0
        for cur_proposals, cur_gt_bboxes, cur_img_meta in zip(proposals, gt_bboxes, img_meta):
            bboxes = []
            cur_proposals = cur_proposals.cpu().numpy()
            for bbox in cur_proposals:
                if slice_num is None:
                    bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
                elif slice_num is not None and slice_num >= math.floor(bbox[4]) and slice_num <= math.ceil(bbox[5]):
                    # select bounding boxes on this slice
                    bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
            part_filename = 'prop' if isProposal else 'anch' 
            filename = 'tests/iter_{}_img_id_{}_batch_{}_{}.png'.format(self.iteration, cur_img_meta['image_id'], batch_num, part_filename)
            self.show_bboxes_gt_bboxes(img, np.array(bboxes), gt_bboxes=cur_gt_bboxes, out_file=filename)
            batch_num += 1

    def visualize_gt_bboxes(self, imgs, gt_bboxes, img_meta):
        gt_bboxes_np = gt_bboxes[0].cpu().numpy()

        for bbox in gt_bboxes_np:
            for slice_num in range(int(bbox[4]), int(bbox[5])):
                img = tensor2img3D(imgs, slice_num=slice_num)
                filename = 'tests/iter_{}_img_id_{}_slice_{}.png'.format(self.iteration, img_meta[0]['image_id'], slice_num)
                mmcv.imshow_bboxes(img, np.array([bbox]), show=False, out_file=filename)
    
    def visualize_gt_bboxes_masks(self, imgs, gt_bboxes, img_meta, gt_masks):
        for num_img in range(len(gt_bboxes)):
            gt_bboxes_np = gt_bboxes[num_img].cpu().numpy()
            gt_masks_np = gt_masks[num_img].cpu().numpy()
            bbox_num = 0
            for bbox in gt_bboxes_np:
                for slice_num in range(int(bbox[4]), int(bbox[5])):
                    img = tensor2img3D(imgs, slice_num=slice_num)
                    filename = 'tests/iter_{}_img_id_{}_bbox_num_{}_slice_{}.png'.format(self.iteration, img_meta[num_img]['image_id'], bbox_num, slice_num)
                    plt.figure()
                    plt.imshow(img)
                    plt.imshow(gt_masks_np[0, slice_num, :, :] * 255, alpha=0.3)
                    ax = plt.gca()
                    rect = pts.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], fill=False, edgecolor='red', linewidth=2)
                    ax.add_patch(rect)
                    plt.savefig(filename)
                    plt.close()
                bbox_num += 1
        breakpoint()

    def imwrite(self, img, file_path, params=None):
        """Write image to file
        """
        return cv2.imwrite(file_path, img, params)

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

    def save_losses_plt(self, losses):
        rpn_cls_loss = torch.sum(torch.stack(losses['loss_rpn_cls'])).item()
        rpn_bbox_reg_loss = torch.sum(torch.stack(losses['loss_rpn_reg'])).item()
        self.rpn_cls_losses.append(rpn_cls_loss)
        self.rpn_bbox_reg_losses.append(rpn_bbox_reg_loss)
        self.total_losses.append(rpn_cls_loss + rpn_bbox_reg_loss)
        self.iterations.append(self.iteration)
        plt.plot(self.iterations, self.rpn_cls_losses, color='skyblue', linewidth=1, label='rpn class loss')
        plt.plot(self.iterations, self.rpn_bbox_reg_losses, color='olive', linewidth=1, label='rpn bbox reg loss')
        plt.plot(self.iterations, self.total_losses, color='red', linewidth=1, label='rpn total loss')
        if self.iteration == 1:
            plt.legend(loc='upper right')
        plt.savefig('tests/iter_{}_loss.png'.format(self.iteration))

        # end of iteration: append new lines to output.json
        # out = open('output.json', 'a+')
        # out.write('\n\n')