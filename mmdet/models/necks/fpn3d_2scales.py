import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from ..utils import ConvModule3D
from ..registry import NECKS


@NECKS.register_module
class FPN3D2Scales(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 input1_size,
                 input2_size,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 normalize=None,
                 activation=None):
        super(FPN3D2Scales, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.with_bias = normalize is None

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        input1_size_test = input1_size.copy()
        input2_size_test = input2_size.copy()
        max_img_scales_arr_index = (len(in_channels))*2 - 1
        self.img_scales = [[] for i in range(max_img_scales_arr_index, -1, -1)]
        self.img_scales_test = [[] for i in range(max_img_scales_arr_index, -1, -1)]
        downscale_factor = 2
        for i in range(max_img_scales_arr_index, 0, -2):
            # initial downscale is 16 for x,y and 2 for z
            if i == max_img_scales_arr_index:
                input2_size[0] /= 2; input2_size[1] /= 16; input2_size[2] /= 16
                input1_size[0] /= 2; input1_size[1] /= 16; input1_size[2] /= 16

                input2_size_test[0] /= 2; input2_size_test[1] /= 4; input2_size_test[2] /= 4
                input1_size_test[0] /= 2; input1_size_test[1] /= 4; input1_size_test[2] /= 4
            else:
                input2_size[0] /= downscale_factor; input2_size[1] /= downscale_factor; input2_size[2] /= downscale_factor
                input1_size[0]/= downscale_factor; input1_size[1] /= downscale_factor; input1_size[2] /= downscale_factor

                input2_size_test[0] /= downscale_factor; input2_size_test[1] /= downscale_factor; input2_size_test[2] /= downscale_factor
                input1_size_test[0]/= downscale_factor; input1_size_test[1] /= downscale_factor; input1_size_test[2] /= downscale_factor

            self.img_scales[i] = [round(input2_size[0]), round(input2_size[1]), round(input2_size[2])]
            self.img_scales[i-1] = [round(input1_size[0]), round(input1_size[1]), round(input1_size[2])]

            self.img_scales_test[i] = [round(input2_size_test[0]), round(input2_size_test[1]), round(input2_size_test[2])]
            self.img_scales_test[i-1] = [round(input1_size_test[0]), round(input1_size_test[1]), round(input1_size_test[2])]

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule3D(
                in_channels[i],
                out_channels,
                1,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)
            l_conv_2 = ConvModule3D(
                in_channels[i],
                out_channels,
                1,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)
            fpn_conv = ConvModule3D(
                out_channels,
                out_channels,
                3,
                padding=1,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)
            fpn_conv_2 = ConvModule3D(
                out_channels,
                out_channels,
                3,
                padding=1,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.lateral_convs.append(l_conv_2)
            self.fpn_convs.append(fpn_conv)
            self.fpn_convs.append(fpn_conv_2)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule3D(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    normalize=normalize,
                    bias=self.with_bias,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule3D
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs, inputs_2, is_test):
        assert len(inputs) == len(self.in_channels)
        assert len(inputs_2) == len(self.in_channels)

        # build laterals
        # laterals = [
        #     lateral_conv(inputs[i + self.start_level])
        #     for i, lateral_conv in enumerate(self.lateral_convs)
        # ]
        # build laterals
        laterals = []
        index1, index2 = 0, 0
        for i, lateral_conv in enumerate(self.lateral_convs):
            if i % 2 == 0:
                laterals.append(lateral_conv(inputs_2[index2 + self.start_level]))
                index2 += 1
            else:
                laterals.append(lateral_conv(inputs[index1 + self.start_level]))
                index1 += 1

        # build top-down path
        used_backbone_levels = len(laterals)
        img_scales_i = 1

        for i in range(used_backbone_levels - 1, 0, -1):
            if is_test:
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=self.img_scales_test[img_scales_i], mode='nearest')
            else:
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=self.img_scales[img_scales_i], mode='nearest')
            img_scales_i += 1

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool3d(outs[-2], 1, stride=2))     
        return tuple(outs)
