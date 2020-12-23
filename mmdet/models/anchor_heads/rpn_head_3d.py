import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.core import delta2bbox3D
from mmdet.ops import nms
from .anchor_head_3d import AnchorHead3D
from ..registry import HEADS


@HEADS.register_module
class RPNHead3D(AnchorHead3D):

    def __init__(self, in_channels, **kwargs):
        super(RPNHead3D, self).__init__(2, in_channels, **kwargs)

    def _init_layers(self):
        self.rpn_conv = nn.Conv3d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv3d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv3d(self.feat_channels, self.num_anchors * 6, 1)

    def init_weights(self):
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)

    def forward_single(self, x):
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             img_metas,
             cfg,
             gt_bboxes_ignore=None,
             iteration=None,
             img_meta_2=None,
             img_meta_3=None):
        if img_meta_2 is not None:
            img_metas = img_meta_2
        if img_meta_3 is not None:
            img_metas = img_meta_3
        losses = super(RPNHead3D, self).loss(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            None,
            img_metas,
            cfg,
            gt_bboxes_ignore=gt_bboxes_ignore,
            iteration=iteration)

        if img_meta_2 is not None:
            return dict(
                loss_rpn_cls_2=losses['loss_cls'], loss_rpn_reg_2=losses['loss_reg'])
        
        if img_meta_3 is not None:
            return dict(
                loss_rpn_cls_3=losses['loss_cls'], loss_rpn_reg_3=losses['loss_reg'])

        return dict(
            loss_rpn_cls=losses['loss_cls'], loss_rpn_reg=losses['loss_reg'])

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        mlvl_proposals = []
        anchors_levels = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-3:] == rpn_bbox_pred.size()[-3:]
            anchors = mlvl_anchors[idx]
            rpn_cls_score = rpn_cls_score.permute(2, 3, 1, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(2, 3, 1, 0).reshape(-1, 6)
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                # filter out all the negative anchors
                if self.pos_indices is not None and self.pos_indices[idx].shape == scores.shape:
                    pos_indices = self.pos_indices[idx]
                    scores = scores[pos_indices]
                    rpn_bbox_pred = rpn_bbox_pred[pos_indices]
                    anchors = anchors[pos_indices]
                elif self.pos_indices_test is not None and self.pos_indices_test[idx].shape == scores.shape:
                    pos_indices = self.pos_indices_test[idx]
                    scores = scores[pos_indices]
                    rpn_bbox_pred = rpn_bbox_pred[pos_indices]
                    anchors = anchors[pos_indices]

                if scores.shape[0] > cfg.nms_pre:
                    _, topk_inds = scores.topk(cfg.nms_pre)
                    rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                    anchors = anchors[topk_inds, :]
                    scores = scores[topk_inds]

                # debug only...
                # out = open('output.json', 'a+')
                # out.write("best anchors.......:\n")
                # out.write("topk_inds: {}\n".format(topk_inds))
                # out.write("anchors: {}\n".format(anchors))
                # out.write("scores: {}\n".format(scores))
                # out.write("num anchors and scores: {}\n".format(len(anchors)))
                # out.write("\n\n")

            proposals = delta2bbox3D(anchors, rpn_bbox_pred, self.target_means,
                                   self.target_stds, img_shape)
            if cfg.min_bbox_size > 0:
                breakpoint()
                w = proposals[:, 2] - proposals[:, 0] + 1
                h = proposals[:, 3] - proposals[:, 1] + 1
                valid_inds = torch.nonzero((w >= cfg.min_bbox_size) &
                                           (h >= cfg.min_bbox_size)).squeeze()
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
            proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)
            anchors_levels.append(anchors)
        anchors_levels = torch.cat(anchors_levels, 0)
        proposals = torch.cat(mlvl_proposals, 0)
        if cfg.nms_across_levels:
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.max_num, :]
        else:
            scores = proposals[:, 6]
            num = min(cfg.max_num, proposals.shape[0])
            # topk_inds = scores > 0.1 # RPN soft cutoff
            _, topk_inds = scores.topk(num) # original code
            proposals = proposals[topk_inds, :]
        return proposals, anchors_levels
