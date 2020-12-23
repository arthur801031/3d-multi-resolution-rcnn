from .rpn_3d import RPN3D
from ..registry import DETECTORS


@DETECTORS.register_module
class RPN3DModule(RPN3D):

    def __init__(self,
                 backbone,
                 rpn_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super(RPN3DModule, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
