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
import os.path
from os import path

@DETECTORS.register_module
class TwoStageDetector3D2ScalesHeads(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 rpn_head=None,
                 rpn_head_2=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStageDetector3D2ScalesHeads, self).__init__()

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

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_roi_extractor_2 = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)
            self.bbox_head_2 = builder.build_head(bbox_head)

        self.mask_head = None; mask_head = None

        if mask_head is not None:
            if mask_roi_extractor is not None:
                self.mask_roi_extractor = builder.build_roi_extractor(
                    mask_roi_extractor)
                # enable second mask head
                self.mask_roi_extractor_2 = builder.build_roi_extractor(
                    mask_roi_extractor)
                self.share_roi_extractor = False
            else:
                breakpoint()
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.bbox_roi_extractor
            self.mask_head = builder.build_head(mask_head)
            # enable second mask head
            self.mask_head_2 = builder.build_head(mask_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(TwoStageDetector3D2ScalesHeads, self).init_weights(pretrained)
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
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_roi_extractor_2.init_weights()
            self.bbox_head.init_weights()
            self.bbox_head_2.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            # enable second mask head
            self.mask_head_2.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()
                # enable second mask head
                self.mask_roi_extractor_2.init_weights()

    def extract_feat(self, imgs):
        x = self.backbone(imgs)
        if self.with_neck:
            x = self.neck(x)
        return x
    
    '''
    Better FPN for 2scales-heads v2
    '''
    def extract_feat_fusion(self, imgs, imgs_2, is_test=False):
        x = self.backbone(imgs)
        x_2 = self.backbone(imgs_2)
        if self.with_neck:
            x_combined = self.neck(x, x_2, is_test)
        x = []
        x_2 = []
        for i, cur_x in enumerate(x_combined):
            if i % 2 == 0:
                x_2.append(cur_x)
            else:
                x.append(cur_x)
        return tuple(x), tuple(x_2)
    
    '''
    Better FPN for 2scales-heads v1
    '''
    def fuse_feature_maps(self, x, x_2):
        upsample_scale = 1.5
        downsample_scale = 1/1.5
        new_x = []
        new_x_2 = []
        for cur_level in range(len(x)):
            # downsample larger feature maps and fuse with lower resolution feature maps
            new_x.append(torch.add(x[cur_level], F.interpolate(x_2[cur_level], scale_factor=downsample_scale, mode='nearest')))
        for cur_level in range(len(x)):
            # upsample smaller feature maps and fuse with higher resolution feature maps
            new_x_2.append(torch.add(x_2[cur_level], F.interpolate(x[cur_level], size=[round(x[cur_level].shape[2] * upsample_scale), round(x[cur_level].shape[3] * upsample_scale), round(x[cur_level].shape[4] * upsample_scale)], mode='nearest')))
        return new_x, new_x_2

    def forward(self, imgs, img_meta, imgs_2, img_meta_2, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(imgs, img_meta, imgs_2, img_meta_2, **kwargs)
        else:
            return self.forward_test(imgs, img_meta,  imgs_2, img_meta_2, **kwargs)

    def forward_train(self,
                      imgs,
                      img_meta,
                      imgs_2,
                      img_meta_2,
                      gt_bboxes,
                      gt_bboxes_2,
                      gt_labels,
                      gt_labels_2,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_masks_2=None,
                      pp=None,
                      pp_2=None,
                      proposals=None):
        # self.print_iterations()
        assert imgs.shape[1] == 3 and imgs_2.shape[1] == 3 # make sure channel size is 3
        # Default FPN
        x = self.extract_feat(imgs)
        x_2 = self.extract_feat(imgs_2)

        ##### WORSE PERFORMANCE
        # Better FPN for 2 scales v1
        # x, x_2 = self.fuse_feature_maps(x, x_2)
        # Better FPN for 2 scales v2
        # x, x_2 = self.extract_feat_fusion(imgs, imgs_2)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_outs_2 = self.rpn_head_2(x_2)

            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_loss_inputs_2 = rpn_outs_2 + (gt_bboxes_2, img_meta,
                                          self.train_cfg.rpn)

            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, iteration=self.iteration)
            rpn_losses_2 = self.rpn_head_2.loss(
                *rpn_loss_inputs_2, gt_bboxes_ignore=gt_bboxes_ignore, iteration=self.iteration, img_meta_2=img_meta_2)

            losses.update(rpn_losses)
            losses.update(rpn_losses_2)

            proposal_inputs = rpn_outs + (img_meta, self.train_cfg.rpn_proposal)
            proposal_inputs_2 = rpn_outs_2 + (img_meta, self.train_cfg.rpn_proposal)

            proposal_list, anchors = self.rpn_head.get_bboxes(*proposal_inputs)
            proposal_list_2, anchors_2 = self.rpn_head_2.get_bboxes(*proposal_inputs_2, img_meta_2=img_meta_2)

            if pp is not None and pp_2 is not None:
                proposal_list = torch.cat((proposal_list[0], pp[0]), 0)
                proposal_list = [proposal_list]
                proposal_list_2 = torch.cat((proposal_list_2[0], pp_2[0]), 0)
                proposal_list_2 = [proposal_list_2]

            # self.rpn_head.visualize_anchor_boxes(imgs, rpn_outs[0], img_meta, slice_num=45, shuffle=True) # debug only
            # self.visualize_proposals(imgs, proposal_list, gt_bboxes, img_meta, slice_num=None, isProposal=True) #debug only
            # self.visualize_proposals(imgs, anchors, gt_bboxes, img_meta, slice_num=None, isProposal=False) #debug only
            # self.visualize_gt_bboxes(imgs, gt_bboxes, img_meta) #debug only
            # breakpoint()
            # self.visualize_gt_bboxes(imgs_2, gt_bboxes_2, img_meta_2) #debug only
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

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi3D([res.bboxes for res in sampling_results])
            rois_2 = bbox2roi3D([res.bboxes for res in sampling_results_2])

            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            bbox_feats_2 = self.bbox_roi_extractor_2(
                x_2[:self.bbox_roi_extractor_2.num_inputs], rois_2)

            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            cls_score_2, bbox_pred_2 = self.bbox_head_2(bbox_feats_2)

            bbox_targets = self.bbox_head.get_target(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)
            bbox_targets_2 = self.bbox_head_2.get_target(
                sampling_results_2, gt_bboxes_2, gt_labels_2, self.train_cfg.rcnn)
            
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            loss_bbox_2 = self.bbox_head_2.loss(cls_score_2, bbox_pred_2,
                                            *bbox_targets_2, img_meta_2=img_meta_2)

            losses.update(loss_bbox)
            losses.update(loss_bbox_2)

        # mask head forward and loss
        if self.with_mask:
            # # implementation #1
            # # only utilize one mask head for higher resolution feature maps
            # pos_rois = bbox2roi3D(
            #     [res.pos_bboxes for res in sampling_results_2])
            # mask_feats = self.mask_roi_extractor(
            #     x_2[:self.mask_roi_extractor.num_inputs], pos_rois)
            # mask_pred = self.mask_head(mask_feats)
            # mask_targets = self.mask_head.get_target(
            #     sampling_results_2, gt_masks, self.train_cfg.rcnn)            
            # pos_labels = torch.cat(
            #     [res.pos_gt_labels for res in sampling_results_2])
            # loss_mask = self.mask_head.loss(mask_pred, mask_targets,
            #                                 pos_labels)
            # losses.update(loss_mask)

            # implementation #2
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

            # higher resolution mask head
            pos_rois = bbox2roi3D(
                [res.pos_bboxes for res in sampling_results_2])
            mask_feats = self.mask_roi_extractor_2(
                x_2[:self.mask_roi_extractor_2.num_inputs], pos_rois)
            mask_pred = self.mask_head_2(mask_feats)
            mask_targets = self.mask_head_2.get_target(
                sampling_results_2, gt_masks_2, self.train_cfg.rcnn)            
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results_2])
            loss_mask_2 = self.mask_head_2.loss(mask_pred, mask_targets,
                                            pos_labels, img_meta_2=img_meta_2)
            losses.update(loss_mask_2)

        # self.save_losses_plt(losses) #debug only...
        self.iteration += 1
        return losses

    def forward_test(self, imgs, img_metas, imgs_2, img_meta_2, **kwargs):
        return self.simple_test(imgs, img_metas, imgs_2, img_meta_2, **kwargs)

    def simple_test(self, imgs, img_metas, imgs_2, img_metas_2, pp=None, pp_2=None, proposals=None, rescale=False, test_cfg2=None):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."
        
        if test_cfg2 is not None:
            test_cfg = test_cfg2
        else:
            test_cfg = self.test_cfg

        img_metas = img_metas[0]
        img_metas_2 = img_metas_2[0]

        # Default FPN
        x = self.extract_feat(imgs)
        x_2 = self.extract_feat(imgs_2)

        #### WORSE PERFORMANCE:
        # Better FPN for 2 scales v1
        # x, x_2 = self.fuse_feature_maps(x, x_2)
        # Better FPN for 2 scales v2
        # x, x_2 = self.extract_feat_fusion(imgs, imgs_2, is_test=True)

        # dataset 1
        proposal_list = self.simple_test_rpn(
            x, img_metas, test_cfg.rpn) if proposals is None else proposals
        # use unsupervised learning's proposals
        if pp is not None:
            proposal_list = torch.cat((proposal_list[0], pp[0]), 0)
            proposal_list = [proposal_list]
        bboxes, scores = self.simple_test_bboxes(
            x, img_metas, proposal_list, None, rescale=rescale)

        # dataset 2
        proposal_list = self.simple_test_rpn_2(
            x_2, img_metas_2, test_cfg.rpn) if proposals is None else proposals
        # use unsupervised learning's proposals
        if pp_2 is not None:
            proposal_list = torch.cat((proposal_list[0], pp_2[0]), 0)
            proposal_list = [proposal_list]
        bboxes_2, scores_2 = self.simple_test_bboxes_2(
            x_2, img_metas_2, proposal_list, None, rescale=rescale)

        bboxes = torch.cat((bboxes, bboxes_2), 0)
        scores = torch.cat((scores, scores_2), 0)
        det_bboxes, det_labels = multiclass_nms_3d(bboxes, scores, test_cfg.rcnn.score_thr, test_cfg.rcnn.nms, test_cfg.rcnn.max_per_img)

        bbox_results = bbox2result3D(det_bboxes, det_labels,
                                        self.bbox_head.num_classes)
        return bbox_results

        # segm_results = self.simple_test_mask(
        #         x, img_metas, det_bboxes, det_labels, rescale=rescale)

        # return bbox_results, segm_results
            
        '''
        bboxes from non-scaled pathway are fed into non-scaled mask branch, while bboxes from up-scaled pathway are 
        fed into upscaled mask branch.
        '''
        # find out which detection box belongs to which resolution
        downscaled_factor = 1.5
        det_bboxes_np = det_bboxes.cpu().numpy()
        det_labels_np = det_labels.cpu().numpy()
        bboxes_np = bboxes.cpu().numpy()
        cutoff_between_res1_res2 = len(bboxes) - len(bboxes_2)
        nonscaled_bboxes = []
        nonscaled_labels = []
        upscaled_bboxes = []
        upscaled_labels = []
        for det_bbox, det_label in zip(det_bboxes_np, det_labels_np):
            for index, bbox in enumerate(bboxes_np):
                if np.all(det_bbox[:6] == bbox[6:]):
                    if index < cutoff_between_res1_res2:
                        # 1x
                        nonscaled_bboxes.append(det_bbox)
                        nonscaled_labels.append(det_label)
                    else:
                        # 1.5x
                        det_bbox_upscaled = det_bbox[:6] / (1/downscaled_factor)
                        det_bbox_upscaled = np.append(det_bbox_upscaled, det_bbox[6])
                        upscaled_bboxes.append(det_bbox_upscaled)
                        upscaled_labels.append(det_label)

        nonscaled_bboxes_gpu = torch.from_numpy(np.array(nonscaled_bboxes)).cuda()
        nonscaled_labels_gpu = torch.from_numpy(np.array(nonscaled_labels)).cuda()
        upscaled_bboxes_gpu = torch.from_numpy(np.array(upscaled_bboxes)).cuda()
        upscaled_labels_gpu = torch.from_numpy(np.array(upscaled_labels)).cuda()

        # replace original scale's ori_shape with upscaled's ori_shape so that mask size is upscaled and correct
        img_metas_2[0]['ori_shape'] = (512, 512, 3)
        img_metas_2[0]['img_shape'] = (512, 512, 3, 240) # full volume only
        # img_metas_2[0]['img_shape'] = (128, 128, 3, 240) # patches only

        segm_results_nonscaled = self.simple_test_mask(
                x, img_metas, nonscaled_bboxes_gpu, nonscaled_labels_gpu, rescale=rescale)

        segm_results_upscaled = self.simple_test_mask_2(
                x_2, img_metas_2, upscaled_bboxes_gpu, upscaled_labels_gpu, rescale=rescale)

        upscaled_bboxes_gpu_downscaled = upscaled_bboxes_gpu[:,:6] / downscaled_factor
        upscaled_bboxes_gpu_downscaled = torch.cat((upscaled_bboxes_gpu_downscaled, upscaled_bboxes_gpu[:,6, None]), dim=1)

        det_bboxes = torch.cat((nonscaled_bboxes_gpu, upscaled_bboxes_gpu_downscaled), 0)
        det_labels = torch.cat((nonscaled_labels_gpu, upscaled_labels_gpu), 0)
        bbox_results = bbox2result3D(det_bboxes, det_labels,
                                        self.bbox_head.num_classes)
        # after this for loop, segm_results_nonscaled contains non-scaled and upscaled segmentation results
        for segm_results in segm_results_upscaled[0]:
            segm_results_nonscaled[0].append(segm_results)

        # for processing full volume:
        return bbox_results, segm_results_nonscaled

        # for processing patch:
        # segm_out_filepath = 'in_progress/segm_results_{}.npz'.format(self.iteration)
        # if not path.exists(segm_out_filepath):
        #     np.savez_compressed(segm_out_filepath, data=segm_results_nonscaled)
        # self.iteration += 1
        # return bbox_results, segm_out_filepath

        
        '''
        bboxes from non-scaled pathway are fed into non-scaled mask branch, while bboxes from up-scaled pathway are 
        fed into upscaled mask branch.
        '''
        # # find out which detection box belongs to which resolution
        # upscaled_factor = 1.5
        # det_bboxes_np = det_bboxes.cpu().numpy()
        # det_labels_np = det_labels.cpu().numpy()
        # bboxes_np = bboxes.cpu().numpy()
        # cutoff_between_res1_res2 = len(bboxes) - len(bboxes_2)
        # nonscaled_bboxes = []
        # nonscaled_labels = []
        # upscaled_bboxes = []
        # upscaled_labels = []
        # for det_bbox, det_label in zip(det_bboxes_np, det_labels_np):
        #     for index, bbox in enumerate(bboxes_np):
        #         if np.all(det_bbox[:6] == bbox[6:]):
        #             if index < cutoff_between_res1_res2:
        #                 # 1x
        #                 det_bbox_downscaled = det_bbox[:6] / upscaled_factor
        #                 det_bbox_downscaled = np.append(det_bbox_downscaled, det_bbox[6])
        #                 nonscaled_bboxes.append(det_bbox_downscaled)
        #                 nonscaled_labels.append(det_label)
        #             else:
        #                 # 1.5x
        #                 upscaled_bboxes.append(det_bbox)
        #                 upscaled_labels.append(det_label)

        # nonscaled_bboxes_gpu = torch.from_numpy(np.array(nonscaled_bboxes)).cuda()
        # nonscaled_labels_gpu = torch.from_numpy(np.array(nonscaled_labels)).cuda()
        # upscaled_bboxes_gpu = torch.from_numpy(np.array(upscaled_bboxes)).cuda()
        # upscaled_labels_gpu = torch.from_numpy(np.array(upscaled_labels)).cuda()

        # # replace original scale's ori_shape with upscaled's ori_shape so that mask size is upscaled and correct
        # # img_metas[0]['img_shape'] = (768, 768, 3, 160) # full volume only
        # img_metas[0]['img_shape'] = (192, 192, 3, 160) # patches only

        # segm_results_nonscaled = self.simple_test_mask(
        #         x, img_metas, nonscaled_bboxes_gpu, nonscaled_labels_gpu, rescale=rescale)

        # segm_results_upscaled = self.simple_test_mask_2(
        #         x_2, img_metas_2, upscaled_bboxes_gpu, upscaled_labels_gpu, rescale=rescale)

        # non_scaled_bboxes_gpu_upscaled = nonscaled_bboxes_gpu[:,:6] / (1/upscaled_factor)
        # non_scaled_bboxes_gpu_upscaled = torch.cat((non_scaled_bboxes_gpu_upscaled, nonscaled_bboxes_gpu[:,6, None]), dim=1)

        # det_bboxes = torch.cat((non_scaled_bboxes_gpu_upscaled, upscaled_bboxes_gpu), 0)
        # det_labels = torch.cat((nonscaled_labels_gpu, upscaled_labels_gpu), 0)
        # bbox_results = bbox2result3D(det_bboxes, det_labels,
        #                                 self.bbox_head.num_classes)
        # # after this for loop, segm_results_nonscaled contains non-scaled and upscaled segmentation results
        # for segm_results in segm_results_upscaled[0]:
        #     segm_results_nonscaled[0].append(segm_results)

        # # for processing full volume:
        # return bbox_results, segm_results_nonscaled

        # # for processing patch:
        # segm_out_filepath = 'in_progress/segm_results_{}.npz'.format(self.iteration)
        # if not path.exists(segm_out_filepath):
        #     np.savez_compressed(segm_out_filepath, data=segm_results_nonscaled)
        # self.iteration += 1
        # return bbox_results, segm_out_filepath

        

        '''
        every bounding box is fed into upscaled mask branch to produce segmentation masks
        '''
        # bbox_results = bbox2result3D(det_bboxes, det_labels,
        #                                 self.bbox_head.num_classes)

        # ############ test RPN's performance ############
        # proposal_list = proposal_list[0].cpu().numpy()
        # return [proposal_list]

        # ############ only return bbox ############
        # return bbox_results

        # if not self.with_mask:
        #     return bbox_results
        # else:
        #     segm_results = self.simple_test_mask_2(
        #         x_2, img_metas_2, det_bboxes, det_labels, rescale=rescale)
        #     # self.visualize_masks_bboxes(bbox_results, segm_results) # debug only

        #     # for full volume
        #     return bbox_results, segm_results

        #     # for processing patch:
        #     # segm_out_filepath = 'in_progress/segm_results_{}.npz'.format(self.iteration)
        #     # if not path.exists(segm_out_filepath):
        #     #     np.savez_compressed(segm_out_filepath, data=segm_results)
        #     # self.iteration += 1
        #     # return bbox_results, segm_out_filepath
            

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