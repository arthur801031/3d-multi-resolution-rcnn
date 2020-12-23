from torch.nn.modules.module import Module
from ..functions.roi_align_3d import RoIAlignFunction3D


class RoIAlign3D(Module):

    def __init__(self, out_size, out_size_depth, spatial_scale, spatial_scale_depth, sample_num=0):
        super(RoIAlign3D, self).__init__()

        self.out_size = out_size
        self.out_size_depth = out_size_depth
        self.spatial_scale = float(spatial_scale)
        self.spatial_scale_depth = float(spatial_scale_depth)
        self.sample_num = int(sample_num)

    def forward(self, features, rois):
        return RoIAlignFunction3D.apply(features, rois, self.out_size, self.out_size_depth,
                                      self.spatial_scale, self.spatial_scale_depth, self.sample_num)
