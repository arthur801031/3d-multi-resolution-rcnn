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
        cls_score, bbox_pred = self.bbox_head(roi_feats)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels

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
    
    def simple_test_bbox_refinement(self,
                                    x,
                                    img_meta,
                                    proposals,
                                    rcnn_test_cfg,
                                    rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi3D(proposals)
        roi_feats = self.bbox_roi_extractor_refinement(
            x[:len(self.bbox_roi_extractor_refinement.featmap_strides)], rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        # regression only
        bbox_pred = self.refinement_head(roi_feats)
        # class and regression
        # _, bbox_pred = self.refinement_head(roi_feats)

        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes = self.refinement_head.get_det_bboxes(
            rois,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes


    def simple_test_bbox_refinement_2(self,
                                    x,
                                    img_meta,
                                    proposals,
                                    rcnn_test_cfg,
                                    rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi3D(proposals)
        roi_feats = self.bbox_roi_extractor_refinement_2(
            x[:len(self.bbox_roi_extractor_refinement_2.featmap_strides)], rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        # regression only
        bbox_pred = self.refinement_head_2(roi_feats)
        # class and regression
        # _, bbox_pred = self.refinement_head(roi_feats)

        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes = self.refinement_head_2.get_det_bboxes(
            rois,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes


    def simple_test_bbox_refinement_3(self,
                                    x,
                                    img_meta,
                                    proposals,
                                    rcnn_test_cfg,
                                    rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi3D(proposals)
        roi_feats = self.bbox_roi_extractor_refinement_3(
            x[:len(self.bbox_roi_extractor_refinement_3.featmap_strides)], rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        # regression only
        bbox_pred = self.refinement_head_3(roi_feats)
        # class and regression
        # _, bbox_pred = self.refinement_head(roi_feats)

        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes = self.refinement_head_3.get_det_bboxes(
            rois,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes


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

            # original code. remove because in two_heads implementation bboxes should not be scaled
            # _bboxes = (det_bboxes[:, :6] * scale_factor if rescale else det_bboxes) 
            _bboxes = det_bboxes[:, :6]

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

            # original code. remove because in two_heads implementation bboxes should not be scaled
            # _bboxes = (det_bboxes[:, :6] * scale_factor if rescale else det_bboxes)
            _bboxes = det_bboxes[:, :6]

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

    def simple_test_mask_refinement(self,
                         x,
                         img_meta,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        # image shape of the first image in the batch (only one)
        scale_factor = img_meta[0]['scale_factor']
        ori_shape = (*img_meta[0]['ori_shape'][:2], int(img_meta[0]['img_shape'][3] / scale_factor))
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head_refinement.num_classes - 1)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.

            # original code. remove because in two_heads implementation bboxes should not be scaled
            # _bboxes = (det_bboxes[:, :6] * scale_factor if rescale else det_bboxes)
            _bboxes = det_bboxes[:, :6]

            mask_rois_refinement = bbox2roi3D([_bboxes])
            mask_feats = self.mask_roi_extractor_refinement(
                x[:len(self.mask_roi_extractor_refinement.featmap_strides)], mask_rois_refinement)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head_refinement(mask_feats)
            segm_result = self.mask_head_refinement.get_seg_masks(
                mask_pred, _bboxes, det_labels, self.test_cfg.rcnn, ori_shape,
                scale_factor, rescale)
        return segm_result

    def simple_test_mask_refinement_v2(self,
                         x,
                         img_meta,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        # image shape of the first image in the batch (only one)
        scale_factor = img_meta[0]['scale_factor']
        ori_shape = (*img_meta[0]['ori_shape'][:2], int(img_meta[0]['img_shape'][3] / scale_factor))
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.refinement_mask_head.num_classes - 1)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.

            # original code. remove because in two_heads implementation bboxes should not be scaled
            # _bboxes = (det_bboxes[:, :6] * scale_factor if rescale else det_bboxes)
            _bboxes = det_bboxes[:, :6]

            mask_rois_refinement = bbox2roi3D([_bboxes])
            mask_feats = self.refinement_mask_roi_extractor(
                x[:len(self.refinement_mask_roi_extractor.featmap_strides)], mask_rois_refinement)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.refinement_mask_head(mask_feats)
            segm_result = self.refinement_mask_head.get_seg_masks(
                mask_pred, _bboxes, det_labels, self.test_cfg.rcnn, ori_shape,
                scale_factor, rescale)
        return segm_result
