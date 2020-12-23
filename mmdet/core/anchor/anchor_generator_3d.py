import torch
import numpy as np
'''
Refer to this issue for this architecture design: https://github.com/open-mmlab/mmdetection/issues/567
'''
class AnchorGenerator3D(object):

    def __init__(self, base_size, scales, depth_scales, ratios, anchor_depth_base, scale_major=True, ctr=None):
        self.base_size = base_size
        self.anchor_depth_base = anchor_depth_base
        self.scales = torch.Tensor(scales)
        self.anchor_depth_scales = torch.Tensor(depth_scales)
        self.ratios = torch.Tensor(ratios)
        self.scale_major = scale_major
        self.ctr = ctr
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        return self.base_anchors.size(0)

    def gen_base_anchors(self):
        w = self.base_size
        h = self.base_size
        z = self.anchor_depth_base
        if self.ctr is None:
            x_ctr = 0.5 * (w - 1)
            y_ctr = 0.5 * (h - 1)
            z_ctr = 0.5 * (z - 1)
        else:
            x_ctr, y_ctr, z_ctr = self.ctr

        h_ratios = torch.sqrt(self.ratios)
        w_ratios = 1 / h_ratios
        z_ratios = h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * self.scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * self.scales[None, :]).view(-1)
            zs = (z * z_ratios[:, None] * self.anchor_depth_scales[None, :]).view(-1)
        else:
            ws = (w * self.scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * self.scales[:, None] * h_ratios[None, :]).view(-1)
            zs = (z * self.anchor_depth_scales[:, None] * z_ratios[None, :]).view(-1)

        base_anchors = torch.stack(
            [
                x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1),
                z_ctr - 0.5 * (zs - 1), z_ctr + 0.5 * (zs - 1)
            ],
            dim=-1).round()

        return base_anchors


    def grid_anchors(self, featmap_size, stride=16, depth_stride=2, device='cuda'):
        base_anchors = self.base_anchors.to(device)
        feat_z, feat_h, feat_w = featmap_size
        shift_x = np.arange(0, feat_w) * stride
        shift_y = np.arange(0, feat_h) * stride
        shift_z = np.arange(0, feat_z) * depth_stride

        shift_xx, shift_yy, shift_zz = np.meshgrid(shift_x, shift_y, shift_z)
        shift_xx = shift_xx.flatten()
        shift_yy = shift_yy.flatten()
        shift_zz = shift_zz.flatten()
        shifts = np.column_stack((shift_xx, shift_yy, shift_xx, shift_yy, shift_zz, shift_zz))
        shifts = torch.from_numpy(shifts).float().to(device)
        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 6)
        return all_anchors

    def valid_flags(self, featmap_size, valid_size, device='cuda'):
        feat_z, feat_h, feat_w = featmap_size
        valid_d, valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w and valid_d <= feat_z
        valid_x = np.zeros(feat_w, dtype=np.uint8)
        valid_y = np.zeros(feat_h, dtype=np.uint8)
        valid_z = np.zeros(feat_z, dtype=np.uint8)

        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_z[:valid_d] = 1

        valid_xx, valid_yy, valid_zz = np.meshgrid(valid_x, valid_y, valid_z)
        valid_xx = torch.from_numpy(valid_xx.flatten()).to(device)
        valid_yy = torch.from_numpy(valid_yy.flatten()).to(device)
        valid_zz = torch.from_numpy(valid_zz.flatten()).to(device)
        valid = valid_xx & valid_yy & valid_zz
        valid = valid[:, None].expand(
            valid.size(0), self.num_base_anchors).contiguous().view(-1)
        return valid
