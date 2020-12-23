from .two_stage_3d_2scales_heads import TwoStageDetector3D2ScalesHeads
from ..registry import DETECTORS


@DETECTORS.register_module
class MaskRCNN3D2ScalesHeads(TwoStageDetector3D2ScalesHeads):

    def __init__(self,
                 backbone,
                 rpn_head,
                 rpn_head_2,
                 bbox_roi_extractor,
                 bbox_head,
                 mask_roi_extractor,
                 mask_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super(MaskRCNN3D2ScalesHeads, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            rpn_head_2=rpn_head_2,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
