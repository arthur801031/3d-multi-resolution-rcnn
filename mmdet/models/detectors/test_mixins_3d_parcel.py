from mmdet.core import (bbox2roi3D, bbox_mapping, merge_aug_proposals,
                        merge_aug_bboxes, merge_aug_masks, multiclass_nms)


class RPNTestMixin(object):

    def simple_test_rpn(self, x, img_meta, rpn_test_cfg):
        rpn_outs = self.rpn_head(x)
        proposal_inputs = rpn_outs + (img_meta, rpn_test_cfg)
        proposal_list, _ = self.rpn_head.get_bboxes(*proposal_inputs)
        return proposal_list
    
    def simple_test_rpn_2(self, x_2, img_meta, rpn_test_cfg):
        rpn_outs = self.rpn_head_2(x_2)
        proposal_inputs = rpn_outs + (img_meta, rpn_test_cfg)
        proposal_list, _ = self.rpn_head_2.get_bboxes(*proposal_inputs)
        return proposal_list
    
    def simple_test_rpn_3(self, x_3, img_meta, rpn_test_cfg):
        rpn_outs = self.rpn_head_3(x_3)
        proposal_inputs = rpn_outs + (img_meta, rpn_test_cfg)
        proposal_list, _ = self.rpn_head_3.get_bboxes(*proposal_inputs)
        return proposal_list

class BBoxTestMixin(object):

    def simple_test_bboxes(self,
                           x,
                           img_meta,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi3D(proposals)
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        cls_score, bbox_pred, parcellation_score = self.bbox_head(roi_feats)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes, det_labels, det_parcellations = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            parcellation_score,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels, det_parcellations

    def simple_test_bboxes_2(self,
                           x,
                           img_meta,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi3D(proposals)
        roi_feats = self.bbox_roi_extractor_2(
            x[:len(self.bbox_roi_extractor_2.featmap_strides)], rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        cls_score, bbox_pred = self.bbox_head_2(roi_feats)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head_2.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels
    
    def simple_test_bboxes_3(self,
                           x,
                           img_meta,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi3D(proposals)
        roi_feats = self.bbox_roi_extractor_3(
            x[:len(self.bbox_roi_extractor_3.featmap_strides)], rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        cls_score, bbox_pred = self.bbox_head_3(roi_feats)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head_3.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels

class MaskTestMixin(object):

    def simple_test_mask(self,
                         x,
                         img_meta,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        # image shape of the first image in the batch (only one)
        scale_factor = img_meta[0]['scale_factor']
        ori_shape = (*img_meta[0]['img_shape'][:2], int(img_meta[0]['img_shape'][3] / scale_factor))
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            _bboxes = (det_bboxes[:, :6] * scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi3D([_bboxes])
            mask_feats = self.mask_roi_extractor(
                x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head(mask_feats)
            segm_result = self.mask_head.get_seg_masks(
                mask_pred, _bboxes, det_labels, self.test_cfg.rcnn, ori_shape,
                scale_factor, rescale)
        return segm_result

    def simple_test_mask_2(self,
                         x,
                         img_meta,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        # image shape of the first image in the batch (only one)
        scale_factor = img_meta[0]['scale_factor']
        ori_shape = (*img_meta[0]['ori_shape'][:2], int(img_meta[0]['img_shape'][3] / scale_factor))
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head_2.num_classes - 1)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            _bboxes = (det_bboxes[:, :6] * scale_factor if rescale else det_bboxes)
            mask_rois_2 = bbox2roi3D([_bboxes])
            mask_feats = self.mask_roi_extractor_2(
                x[:len(self.mask_roi_extractor_2.featmap_strides)], mask_rois_2)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head_2(mask_feats)
            segm_result = self.mask_head_2.get_seg_masks(
                mask_pred, _bboxes, det_labels, self.test_cfg.rcnn, ori_shape,
                scale_factor, rescale)
        return segm_result
