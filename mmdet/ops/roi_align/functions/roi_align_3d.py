from torch.autograd import Function

from .. import roi_align_cuda


class RoIAlignFunction3D(Function):

    @staticmethod
    def forward(ctx, features, rois, out_size, out_size_depth, spatial_scale, spatial_scale_depth, sample_num=0):
        if isinstance(out_size, int):
            out_h = out_size
            out_w = out_size
            out_d = out_size_depth
        elif isinstance(out_size, tuple):
            breakpoint()
            assert len(out_size) == 2
            assert isinstance(out_size[0], int)
            assert isinstance(out_size[1], int)
            out_h, out_w = out_size
        else:
            raise TypeError(
                '"out_size" must be an integer or tuple of integers')
        ctx.spatial_scale = spatial_scale
        ctx.spatial_scale_depth = spatial_scale_depth
        ctx.sample_num = sample_num
        ctx.save_for_backward(rois)
        ctx.feature_size = features.size()

        batch_size, num_channels, data_depth, data_height, data_width = features.size()
        num_rois = rois.size(0)

        output = features.new_zeros(num_rois, num_channels, out_d, out_h, out_w)
        if features.is_cuda:
            roi_align_cuda.forward3d(features, rois, out_d, out_h, out_w, spatial_scale, spatial_scale_depth,
                                   sample_num, output)
        else:
            raise NotImplementedError
        return output

    @staticmethod
    def backward3d_test(top_grad, rois, pooled_depth, pooled_height, pooled_width,
                        spatial_scale, sample_num, bottom_grad):
        # declared inside c++ file
        num_rois = rois.size(0)
        num_channels = bottom_grad.size(1)
        data_depth = bottom_grad.size(2)
        data_height = bottom_grad.size(3)
        data_width = bottom_grad.size(4)
        channels = num_channels
        depth = data_depth

        # CUDA file
        output_size = num_rois * pooled_depth * pooled_height * pooled_width * channels
        top_diff = top_grad.data
        rois_data = rois.data
        bottom_diff = bottom_grad.data

        index = 66664227
        pw = int(index % pooled_width)
        ph = int((index / pooled_width) % pooled_height)
        pd = int((index / pooled_width / pooled_height) % pooled_depth)
        n = int(index / pooled_width / pooled_height / pooled_depth / channels)
        
        offset_top = (n * depth + pd) * pooled_height * pooled_width + ph * pooled_width + pw

        ForkedPdb().set_trace()

    @staticmethod
    def backward(ctx, grad_output):
        feature_size = ctx.feature_size
        spatial_scale = ctx.spatial_scale
        spatial_scale_depth = ctx.spatial_scale_depth
        sample_num = ctx.sample_num
        rois = ctx.saved_tensors[0]
        assert (feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_depth, data_height, data_width = feature_size
        out_w = grad_output.size(4)
        out_h = grad_output.size(3)
        out_d = grad_output.size(2)

        grad_input = grad_rois = None
        if ctx.needs_input_grad[0]:
            grad_input = rois.new_zeros(batch_size, num_channels, data_depth, data_height,
                                        data_width)
            roi_align_cuda.backward3d(grad_output.contiguous(), rois, out_d, out_h,
                                    out_w, spatial_scale, spatial_scale_depth, sample_num,
                                    grad_input)
        return grad_input, grad_rois, None, None, None, None, None


roi_align = RoIAlignFunction3D.apply
