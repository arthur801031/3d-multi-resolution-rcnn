import torch
import torch.nn as nn

from .base import BaseDetector
from .test_mixins_3d import RPNTestMixin, BBoxTestMixin, MaskTestMixin
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2roi3D, bbox2result3D, build_assigner, build_sampler, tensor2img3D
import mmcv
import math
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as pts
import numpy as np


@DETECTORS.register_module
class RPN3D(BaseDetector, RPNTestMixin, BBoxTestMixin,
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
        super(RPN3D, self).__init__()

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

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(RPN3D, self).init_weights(pretrained)
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

    def extract_feat(self, imgs):
        x = self.backbone(imgs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, imgs, img_meta, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(imgs, img_meta, **kwargs)
        else:
            return self.forward_test(imgs, img_meta, **kwargs)

    def forward_train(self,
                      imgs,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        # self.print_iterations()
        assert imgs.shape[1] == 3 # make sure channel size is 3
        x = self.extract_feat(imgs)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, iteration=self.iteration)
            losses.update(rpn_losses)

            proposal_inputs = rpn_outs + (img_meta, self.train_cfg.rpn_proposal)
            proposal_list, anchors = self.rpn_head.get_bboxes(*proposal_inputs)
            # self.rpn_head.visualize_anchor_boxes(imgs, rpn_outs[0], img_meta, slice_num=45, shuffle=True) # debug only
            # self.visualize_proposals(imgs, proposal_list, gt_bboxes, img_meta, slice_num=None, isProposal=True) #debug only
            # self.visualize_proposals(imgs, anchors, gt_bboxes, img_meta, slice_num=None, isProposal=False) #debug only
            # self.visualize_gt_bboxes(imgs, gt_bboxes, img_meta) #debug only
            # self.visualize_gt_bboxes_masks(imgs, gt_bboxes, img_meta, gt_masks) # debug only
        else:
            proposal_list = proposals

        # self.save_losses_plt(losses) #debug only...
        self.iteration += 1
        return losses

    def forward_test(self, imgs, img_metas, **kwargs):
        return self.simple_test(imgs, img_metas, **kwargs)

    def simple_test(self, imgs, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        img_metas = img_metas[0]
        x = self.extract_feat(imgs)

        proposal_list = self.simple_test_rpn(
            x, img_metas, self.test_cfg.rpn) if proposals is None else proposals
        
        ############ test RPN's performance ############
        proposal_list = proposal_list[0].cpu().numpy()
        return [proposal_list]


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