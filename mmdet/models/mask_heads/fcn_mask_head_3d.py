import mmcv
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn

from ..registry import HEADS
from ..utils import ConvModule3D
from mmdet.core import mask_cross_entropy, mask_target
from skimage.transform import resize
import matplotlib.pyplot as plt


@HEADS.register_module
class FCNMaskHead3D(nn.Module):

    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 upsample_method='deconv',
                 upsample_ratio=2,
                 num_classes=81,
                 class_agnostic=False,
                 normalize=None):
        super(FCNMaskHead3D, self).__init__()
        if upsample_method not in [None, 'deconv', 'nearest', 'bilinear']:
            raise ValueError(
                'Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear"'.format(upsample_method))
        self.num_convs = num_convs
        self.roi_feat_size = roi_feat_size  # WARN: not used and reserved
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = upsample_method
        self.upsample_ratio = upsample_ratio
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.normalize = normalize
        self.with_bias = normalize is None

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (self.in_channels
                           if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule3D(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    normalize=normalize,
                    bias=self.with_bias))
        upsample_in_channels = (self.conv_out_channels
                                if self.num_convs > 0 else in_channels)
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            self.upsample = nn.ConvTranspose3d(
                upsample_in_channels,
                self.conv_out_channels,
                self.upsample_ratio,
                stride=self.upsample_ratio)
        else:
            self.upsample = nn.Upsample(
                scale_factor=self.upsample_ratio, mode=self.upsample_method)

        out_channels = 1 if self.class_agnostic else self.num_classes
        logits_in_channel = (self.conv_out_channels
                             if self.upsample_method == 'deconv' else
                             upsample_in_channels)
        self.conv_logits = nn.Conv3d(logits_in_channel, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None


    def init_weights(self):
        for m in [self.upsample, self.conv_logits]:
            if m is None:
                continue
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        mask_pred = self.conv_logits(x)
        return mask_pred

    def get_target(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets

    def loss(self, mask_pred, mask_targets, labels, img_meta_2=None, img_meta_3=None, img_meta_refinement=None):
        loss = dict()
        if self.class_agnostic:
            loss_mask = mask_cross_entropy(mask_pred, mask_targets,
                                           torch.zeros_like(labels))
        else:
            loss_mask = mask_cross_entropy(mask_pred, mask_targets, labels)
        
        if img_meta_refinement is not None:
            loss['loss_mask_refinement'] = loss_mask
        elif img_meta_3 is not None:
            loss['loss_mask_3'] = loss_mask
        elif img_meta_2 is not None:
            loss['loss_mask_2'] = loss_mask
        else:
            loss['loss_mask'] = loss_mask
        return loss

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class+1, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid().cpu().numpy()
        assert isinstance(mask_pred, np.ndarray)

        cls_segms = [[] for _ in range(self.num_classes - 1)]
        bboxes = det_bboxes.cpu().numpy()[:, :6]
        labels = det_labels.cpu().numpy() + 1

        if rescale:
            img_h, img_w, img_d = ori_shape[:3]
        else:
            breakpoint()
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        # # scale down to match full volume size 
        # img_h /= 2; img_h = round(img_h)
        # img_w /= 2; img_w = round(img_w)
        for i in range(bboxes.shape[0]):
            # original code
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            
            # # scale down to match full volume size 
            # bbox_tmp = (bboxes[i, :4] / 2).astype(np.int32)
            # bbox = np.concatenate((bbox_tmp, (bboxes[i, 4:6]).astype(np.int32)), axis=0)
            
            label = labels[i]
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)
            d = max(bbox[5] - bbox[4] + 1, 1)

            if not self.class_agnostic:
                mask_pred_ = mask_pred[i, label, :, :, :]
            else:
                mask_pred_ = mask_pred[i, 0, :, :, :]

            # resize mask_pred depth 
            im_mask = np.zeros((img_d, img_h, img_w), dtype=np.uint8)
            # im_mask = np.zeros((160, 512, 512), dtype=np.uint8) # fixed scaling...
            bbox_mask = resize(mask_pred_, (d, h, w))
            bbox_mask = (bbox_mask > rcnn_test_cfg.mask_thr_binary).astype(np.uint8)
            im_mask[bbox[4]:bbox[4] + d, bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask
            # self.plt_mask(im_mask); breakpoint() # debug only

            cls_segms[label - 1].append(im_mask)

        return cls_segms

    def plt_mask(self, mask):
        for cur_slice in range(mask.shape[0]):
            filename = 'tests/predicted_mask_{}.png'.format(cur_slice)
            plt.figure()
            plt.imshow(mask[cur_slice,:,:])
            plt.savefig(filename)
            plt.close()