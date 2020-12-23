#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

// #define THREADS_PER_BLOCK 1024
#define THREADS_PER_BLOCK 512

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = 65000;
  return min(optimal_block_num, max_block_num);
}

template <typename scalar_t>
__device__ scalar_t bilinear_interpolate(const scalar_t *bottom_data,
                                         const int height, const int width,
                                         scalar_t y, scalar_t x) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    return 0;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (scalar_t)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (scalar_t)x_low;
  } else {
    x_high = x_low + 1;
  }

  scalar_t ly = y - y_low;
  scalar_t lx = x - x_low;
  scalar_t hy = 1. - ly;
  scalar_t hx = 1. - lx;
  // do bilinear interpolation
  scalar_t lt = bottom_data[y_low * width + x_low];
  scalar_t rt = bottom_data[y_low * width + x_high];
  scalar_t lb = bottom_data[y_high * width + x_low];
  scalar_t rb = bottom_data[y_high * width + x_high];
  scalar_t w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  scalar_t val = (w1 * lt + w2 * rt + w3 * lb + w4 * rb);

  return val;
}

template <typename scalar_t>
__device__ scalar_t bilinear_interpolate_3d(const scalar_t *bottom_data,
                                         const int depth, const int height, const int width,
                                         scalar_t z, scalar_t y, scalar_t x) {
  // deal with cases that inverse elements are out of feature map boundary
  if (z < -1.0 || z > depth || y < -1.0 || y > height || x < -1.0 || x > width) {
    return 0;
  }

  if (z <= 0) z = 0;
  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int z_low = (int)z;
  int y_low = (int)y;
  int x_low = (int)x;
  int z_high;
  int y_high;
  int x_high;

  if (z_low >= depth - 1) {
    z_high = z_low = depth - 1;
    z = (scalar_t)z_low;
  } else {
    z_high = z_low + 1;
  }

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (scalar_t)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (scalar_t)x_low;
  } else {
    x_high = x_low + 1;
  }

  scalar_t lz = z - z_low;
  scalar_t ly = y - y_low;
  scalar_t lx = x - x_low;
  scalar_t hz = 1. - lz;
  scalar_t hy = 1. - ly;
  scalar_t hx = 1. - lx;

  // based on https://github.com/pfjaeger/medicaldetectiontoolkit/blob/master/cuda_functions/roi_align_3D/roi_align/src/cuda/crop_and_resize_kernel.cu
  // do bilinear interpolation

  const float top_left_front = bottom_data[x_low + width * (y_low + height * z_low)];
  const float top_right_front = bottom_data[x_high + width * (y_low + height * z_low)];
  const float bottom_left_front = bottom_data[x_low + width * (y_high + height * z_low)];
  const float bottom_right_front = bottom_data[x_high + width * (y_high + height * z_low)];
  const float top_left_back = bottom_data[x_low + width * (y_low + height * z_high)];
  const float top_right_back = bottom_data[x_high + width * (y_low + height * z_high)];
  const float bottom_left_back = bottom_data[x_low + width * (y_high + height * z_high)];
  const float bottom_right_back = bottom_data[x_high + width * (y_high + height * z_high)];

  // previous implementation - incorrect
  // const float top_left_front = bottom_data[z_low + depth * (x_low + width * y_low)];
  // const float top_right_front = bottom_data[z_low + depth * (x_high + width * y_low)];
  // const float bottom_left_front = bottom_data[z_low + depth * (x_low + width * y_high)];
  // const float bottom_right_front = bottom_data[z_low + depth * (x_high + width * y_high)];
  // const float top_left_back = bottom_data[z_high + depth * (x_low + width * y_low)];
  // const float top_right_back = bottom_data[z_high + depth * (x_high + width * y_low)];
  // const float bottom_left_back = bottom_data[z_high + depth * (x_low + width * y_high)];
  // const float bottom_right_back = bottom_data[z_high + depth * (x_high + width * y_high)];

  scalar_t w1 = hx * hy * hz; scalar_t w5 = hx * hy * lz;
  scalar_t w2 = lx * hy * hz; scalar_t w6 = lx * hy * lz;
  scalar_t w3 = hx * ly * hz; scalar_t w7 = hx * ly * lz;
  scalar_t w4 = lx * ly * hz; scalar_t w8 = lx * ly * lz;

  scalar_t val = (w1 * top_left_front +
                  w2 * top_right_front + 
                  w3 * bottom_left_front +
                  w4 * bottom_right_front +
                  w5 * top_left_back +
                  w6 * top_right_back +
                  w7 * bottom_left_back +
                  w8 * bottom_right_back);

  return val;
}

template <typename scalar_t>
__global__ void ROIAlignForward(const int nthreads, const scalar_t *bottom_data,
                                const scalar_t *bottom_rois,
                                const scalar_t spatial_scale,
                                const int sample_num, const int channels,
                                const int height, const int width,
                                const int pooled_height, const int pooled_width,
                                scalar_t *top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the aligned output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const scalar_t *offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    scalar_t roi_start_w = offset_bottom_rois[1] * spatial_scale;
    scalar_t roi_start_h = offset_bottom_rois[2] * spatial_scale;
    scalar_t roi_end_w = (offset_bottom_rois[3] + 1) * spatial_scale;
    scalar_t roi_end_h = (offset_bottom_rois[4] + 1) * spatial_scale;

    // Force malformed ROIs to be 1x1
    scalar_t roi_width = fmaxf((scalar_t)roi_end_w - roi_start_w, 0.);
    scalar_t roi_height = fmaxf((scalar_t)roi_end_h - roi_start_h, 0.);

    scalar_t bin_size_h = roi_height / pooled_height;
    scalar_t bin_size_w = roi_width / pooled_width;

    const scalar_t *offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * height * width;

    int sample_num_h = (sample_num > 0)
                           ? sample_num
                           : ceil(roi_height / pooled_height);  // e.g., = 2
    int sample_num_w =
        (sample_num > 0) ? sample_num : ceil(roi_width / pooled_width);

    scalar_t h = (scalar_t)(ph + 0.5) * bin_size_h + roi_start_h;
    scalar_t w = (scalar_t)(pw + 0.5) * bin_size_w + roi_start_w;

    int hstart = fminf(floor(h), height - 2);
    int wstart = fminf(floor(w), width - 2);

    scalar_t output_val = 0;
    for (int iy = 0; iy < sample_num_h; iy++) {
      const scalar_t y = roi_start_h + ph * bin_size_h +
                         (scalar_t)(iy + scalar_t(.5f)) * bin_size_h /
                             (scalar_t)(sample_num_h);
      for (int ix = 0; ix < sample_num_w; ix++) {
        const scalar_t x = roi_start_w + pw * bin_size_w +
                           (scalar_t)(ix + scalar_t(.5f)) * bin_size_w /
                               (scalar_t)(sample_num_w);
        scalar_t val = bilinear_interpolate<scalar_t>(offset_bottom_data,
                                                      height, width, y, x);
        output_val += val;
      }
    }
    output_val /= (sample_num_h * sample_num_w);
    top_data[index] = output_val;
  }
}

template <typename scalar_t>
__global__ void ROIAlignForward3D(const int nthreads, const scalar_t *bottom_data,
                                const scalar_t *bottom_rois,
                                const scalar_t spatial_scale,
                                const scalar_t spatial_scale_depth,
                                const int sample_num, const int channels,
                                const int depth, const int height, const int width,
                                const int pooled_depth, const int pooled_height, const int pooled_width,
                                scalar_t *top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, depth, ph, pw) is an element in the aligned output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int pd = (index / pooled_width / pooled_height) % pooled_depth;
    int c = (index / pooled_width / pooled_height / pooled_depth) % channels;
    int n = index / pooled_width / pooled_height / pooled_depth / channels;

    const scalar_t *offset_bottom_rois = bottom_rois + n * 7;
    int roi_batch_ind = offset_bottom_rois[0];
    scalar_t roi_start_w = offset_bottom_rois[1] * spatial_scale;
    scalar_t roi_start_h = offset_bottom_rois[2] * spatial_scale;
    scalar_t roi_end_w = (offset_bottom_rois[3] + 1) * spatial_scale;
    scalar_t roi_end_h = (offset_bottom_rois[4] + 1) * spatial_scale;
    scalar_t roi_start_d = offset_bottom_rois[5] * spatial_scale_depth;
    scalar_t roi_end_d = (offset_bottom_rois[6] + 1) * spatial_scale_depth;

    // Force malformed ROIs to be 1x1
    scalar_t roi_width = fmaxf((scalar_t)roi_end_w - roi_start_w, 0.);
    scalar_t roi_height = fmaxf((scalar_t)roi_end_h - roi_start_h, 0.);
    scalar_t roi_depth = fmaxf((scalar_t)roi_end_d - roi_start_d, 0.);

    scalar_t bin_size_d = roi_depth / pooled_depth;
    scalar_t bin_size_h = roi_height / pooled_height;
    scalar_t bin_size_w = roi_width / pooled_width;

    const scalar_t *offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * depth * height * width;

    int sample_num_d =
        (sample_num > 0) ? sample_num : ceil(roi_depth / pooled_depth);

    int sample_num_h = (sample_num > 0)
                           ? sample_num
                           : ceil(roi_height / pooled_height);  // e.g., = 2
    int sample_num_w =
        (sample_num > 0) ? sample_num : ceil(roi_width / pooled_width);

    scalar_t d = (scalar_t)(pd + 0.5) * bin_size_d + roi_start_d;
    scalar_t h = (scalar_t)(ph + 0.5) * bin_size_h + roi_start_h;
    scalar_t w = (scalar_t)(pw + 0.5) * bin_size_w + roi_start_w;

    int dstart = fminf(floor(d), depth - 2);
    int hstart = fminf(floor(h), height - 2);
    int wstart = fminf(floor(w), width - 2);

    scalar_t output_val = 0;
    for (int iz = 0; iz < sample_num_d; iz++) {
      const scalar_t z = roi_start_d + pd * bin_size_d +
                         (scalar_t)(iz + scalar_t(.5f)) * bin_size_d /
                             (scalar_t)(sample_num_d);
      for (int iy = 0; iy < sample_num_h; iy++) {
        const scalar_t y = roi_start_h + ph * bin_size_h +
                          (scalar_t)(iy + scalar_t(.5f)) * bin_size_h /
                              (scalar_t)(sample_num_h);
        for (int ix = 0; ix < sample_num_w; ix++) {
          const scalar_t x = roi_start_w + pw * bin_size_w +
                            (scalar_t)(ix + scalar_t(.5f)) * bin_size_w /
                                (scalar_t)(sample_num_w);
          scalar_t val = bilinear_interpolate_3d<scalar_t>(offset_bottom_data,
                                                        depth, height, width, z, y, x);
          output_val += val;
        }
      }
    }
    output_val /= (sample_num_d * sample_num_h * sample_num_w);
    top_data[index] = output_val;
  }
}

int ROIAlignForwardLaucher(const at::Tensor features, const at::Tensor rois,
                           const float spatial_scale, const int sample_num,
                           const int channels, const int height,
                           const int width, const int num_rois,
                           const int pooled_height, const int pooled_width,
                           at::Tensor output) {
  const int output_size = num_rois * pooled_height * pooled_width * channels;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      features.type(), "ROIAlignLaucherForward", ([&] {
        const scalar_t *bottom_data = features.data<scalar_t>();
        const scalar_t *rois_data = rois.data<scalar_t>();
        scalar_t *top_data = output.data<scalar_t>();

        ROIAlignForward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, bottom_data, rois_data, scalar_t(spatial_scale),
                sample_num, channels, height, width, pooled_height,
                pooled_width, top_data);
      }));
  THCudaCheck(cudaGetLastError());
  return 1;
}

int ROIAlignForwardLaucher3D(const at::Tensor features, const at::Tensor rois,
                             const float spatial_scale, const float spatial_scale_depth, const int sample_num,
                             const int channels, const int depth, const int height,
                             const int width, const int num_rois,
                             const int pooled_depth, const int pooled_height, const int pooled_width,
                             at::Tensor output) {
  const int output_size = num_rois * pooled_depth * pooled_height * pooled_width * channels;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      features.type(), "ROIAlignLaucherForward3D", ([&] {
        const scalar_t *bottom_data = features.data<scalar_t>();
        const scalar_t *rois_data = rois.data<scalar_t>();
        scalar_t *top_data = output.data<scalar_t>();

        ROIAlignForward3D<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, bottom_data, rois_data, scalar_t(spatial_scale), scalar_t(spatial_scale_depth),
                sample_num, channels, depth, height, width, pooled_depth, pooled_height,
                pooled_width, top_data);
      }));
  THCudaCheck(cudaGetLastError());
  return 1;
}

template <typename scalar_t>
__device__ void bilinear_interpolate_gradient(const int height, const int width,
                                              scalar_t y, scalar_t x,
                                              scalar_t &w1, scalar_t &w2,
                                              scalar_t &w3, scalar_t &w4,
                                              int &x_low, int &x_high,
                                              int &y_low, int &y_high) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  y_low = (int)y;
  x_low = (int)x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (scalar_t)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (scalar_t)x_low;
  } else {
    x_high = x_low + 1;
  }

  scalar_t ly = y - y_low;
  scalar_t lx = x - x_low;
  scalar_t hy = 1. - ly;
  scalar_t hx = 1. - lx;

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}

template <typename scalar_t>
__device__ void bilinear_interpolate_gradient_3d(const int depth, const int height, const int width,
                                              scalar_t z, scalar_t y, scalar_t x,
                                              scalar_t &w1, scalar_t &w2,
                                              scalar_t &w3, scalar_t &w4,
                                              scalar_t &w5, scalar_t &w6,
                                              scalar_t &w7, scalar_t &w8,
                                              int &x_low, int &x_high,
                                              int &y_low, int &y_high,
                                              int &z_low, int &z_high) {
  // deal with cases that inverse elements are out of feature map boundary
  if (z < -1.0 || z > depth || y < -1.0 || y > height || x < -1.0 || x > width) {
    w1 = w2 = w3 = w4 = w5 = w6 = w7 = w8 = 0.;
    x_low = x_high = y_low = y_high = z_low = z_high = -1;
    return;
  }

  if (z <= 0) z = 0;
  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  z_low = (int)z;
  y_low = (int)y;
  x_low = (int)x;

  if (z_low >= depth - 1) {
    z_high = z_low = depth - 1;
    z = (scalar_t)z_low;
  } else {
    z_high = z_low + 1;
  }

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (scalar_t)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (scalar_t)x_low;
  } else {
    x_high = x_low + 1;
  }

  scalar_t lz = z - z_low;
  scalar_t ly = y - y_low;
  scalar_t lx = x - x_low;
  scalar_t hz = 1. - lz;
  scalar_t hy = 1. - ly;
  scalar_t hx = 1. - lx;

  w1 = hx * hy * hz; w5 = hx * hy * lz;
  w2 = lx * hy * hz; w6 = lx * hy * lz;
  w3 = hx * ly * hz; w7 = hx * ly * lz;
  w4 = lx * ly * hz; w8 = lx * ly * lz;

  return;
}

template <typename scalar_t>
__global__ void ROIAlignBackward(
    const int nthreads, const scalar_t *top_diff, const scalar_t *bottom_rois,
    const scalar_t spatial_scale, const int sample_num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, scalar_t *bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the aligned output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const scalar_t *offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    scalar_t roi_start_w = offset_bottom_rois[1] * spatial_scale;
    scalar_t roi_start_h = offset_bottom_rois[2] * spatial_scale;
    scalar_t roi_end_w = (offset_bottom_rois[3] + 1) * spatial_scale;
    scalar_t roi_end_h = (offset_bottom_rois[4] + 1) * spatial_scale;

    // Force malformed ROIs to be 1x1
    scalar_t roi_width = fmaxf((scalar_t)roi_end_w - roi_start_w, 0.);
    scalar_t roi_height = fmaxf((scalar_t)roi_end_h - roi_start_h, 0.);

    scalar_t bin_size_h = roi_height / pooled_height;
    scalar_t bin_size_w = roi_width / pooled_width;

    scalar_t *offset_bottom_diff =
        bottom_diff + (roi_batch_ind * channels + c) * height * width;
    int offset_top = (n * channels + c) * pooled_height * pooled_width +
                     ph * pooled_width + pw;
    scalar_t offset_top_diff = top_diff[offset_top];

    int sample_num_h = (sample_num > 0)
                           ? sample_num
                           : ceil(roi_height / pooled_height);  // e.g., = 2
    int sample_num_w =
        (sample_num > 0) ? sample_num : ceil(roi_width / pooled_width);

    const scalar_t count = (scalar_t)(sample_num_h * sample_num_w);

    scalar_t h = (scalar_t)(ph + 0.5) * bin_size_h + roi_start_h;
    scalar_t w = (scalar_t)(pw + 0.5) * bin_size_w + roi_start_w;

    int hstart = fminf(floor(h), height - 2);
    int wstart = fminf(floor(w), width - 2);

    for (int iy = 0; iy < sample_num_h; iy++) {
      const scalar_t y =
          roi_start_h + ph * bin_size_h +
          (scalar_t)(iy + .5f) * bin_size_h / (scalar_t)(sample_num_h);
      for (int ix = 0; ix < sample_num_w; ix++) {
        const scalar_t x =
            roi_start_w + pw * bin_size_w +
            (scalar_t)(ix + .5f) * bin_size_w / (scalar_t)(sample_num_w);
        scalar_t w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient<scalar_t>(
            height, width, y, x, w1, w2, w3, w4, x_low, x_high, y_low, y_high);
        scalar_t g1 = offset_top_diff * w1 / count;
        scalar_t g2 = offset_top_diff * w2 / count;
        scalar_t g3 = offset_top_diff * w3 / count;
        scalar_t g4 = offset_top_diff * w4 / count;
        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          atomicAdd(offset_bottom_diff + y_low * width + x_low, g1);
          atomicAdd(offset_bottom_diff + y_low * width + x_high, g2);
          atomicAdd(offset_bottom_diff + y_high * width + x_low, g3);
          atomicAdd(offset_bottom_diff + y_high * width + x_high, g4);
        }
      }
    }
  }
}

template <typename scalar_t>
__global__ void ROIAlignBackward3D(
    const int nthreads, const scalar_t *top_diff, const scalar_t *bottom_rois,
    const scalar_t spatial_scale, const scalar_t spatial_scale_depth, const int sample_num, const int channels,
    const int depth, const int height, const int width, const int pooled_depth, const int pooled_height,
    const int pooled_width, scalar_t *bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, pd, ph, pw) is an element in the aligned output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int pd = (index / pooled_width / pooled_height) % pooled_depth;
    int c = (index / pooled_width / pooled_height / pooled_depth) % channels;
    int n = index / pooled_width / pooled_height / pooled_depth / channels;

    const scalar_t *offset_bottom_rois = bottom_rois + n * 7;
    int roi_batch_ind = offset_bottom_rois[0];
    scalar_t roi_start_w = offset_bottom_rois[1] * spatial_scale;
    scalar_t roi_start_h = offset_bottom_rois[2] * spatial_scale;
    scalar_t roi_end_w = (offset_bottom_rois[3] + 1) * spatial_scale;
    scalar_t roi_end_h = (offset_bottom_rois[4] + 1) * spatial_scale;
    scalar_t roi_start_d = offset_bottom_rois[5] * spatial_scale_depth;
    scalar_t roi_end_d = (offset_bottom_rois[6] + 1) * spatial_scale_depth;

    // Force malformed ROIs to be 1x1
    scalar_t roi_width = fmaxf((scalar_t)roi_end_w - roi_start_w, 0.);
    scalar_t roi_height = fmaxf((scalar_t)roi_end_h - roi_start_h, 0.);
    scalar_t roi_depth = fmaxf((scalar_t)roi_end_d - roi_start_d, 0.);

    scalar_t bin_size_h = roi_height / pooled_height;
    scalar_t bin_size_w = roi_width / pooled_width;
    scalar_t bin_size_d = roi_depth / pooled_depth;

    scalar_t *offset_bottom_diff =
      bottom_diff + (roi_batch_ind * channels + c) * depth * height * width;

    int offset_top = (n * channels + c) * pooled_depth * pooled_height * pooled_width + 
      pd * pooled_depth * pooled_width + ph * pooled_width + pw;

    scalar_t offset_top_diff = top_diff[offset_top];
    
    // another impplementation (incorrect)
    // int top_offset = (n * channels + c) * pooled_depth * pooled_height * pooled_width;
    // const scalar_t *offset_top_diff_tmp = top_diff + top_offset;
    // scalar_t offset_top_diff = offset_top_diff_tmp[pd * pooled_depth * pooled_height + ph * pooled_width + pw];

    int sample_num_h = (sample_num > 0)
                           ? sample_num
                           : ceil(roi_height / pooled_height);  // e.g., = 2
    int sample_num_w =
        (sample_num > 0) ? sample_num : ceil(roi_width / pooled_width);
    
    int sample_num_d =
        (sample_num > 0) ? sample_num : ceil(roi_depth / pooled_depth);

    const scalar_t count = (scalar_t)(sample_num_d * sample_num_h * sample_num_w);

    scalar_t h = (scalar_t)(ph + 0.5) * bin_size_h + roi_start_h;
    scalar_t w = (scalar_t)(pw + 0.5) * bin_size_w + roi_start_w;
    scalar_t d = (scalar_t)(pd + 0.5) * bin_size_d + roi_start_d;

    int hstart = fminf(floor(h), height - 2);
    int wstart = fminf(floor(w), width - 2);
    int dstart = fminf(floor(d), depth - 2);

    for (int iz = 0; iz < sample_num_d; iz++) {
      const scalar_t z =
          roi_start_d + pd * bin_size_d +
          (scalar_t)(iz + .5f) * bin_size_d / (scalar_t)(sample_num_d);
      for (int iy = 0; iy < sample_num_h; iy++) {
        const scalar_t y =
            roi_start_h + ph * bin_size_h +
            (scalar_t)(iy + .5f) * bin_size_h / (scalar_t)(sample_num_h);
        for (int ix = 0; ix < sample_num_w; ix++) {
          const scalar_t x =
              roi_start_w + pw * bin_size_w +
              (scalar_t)(ix + .5f) * bin_size_w / (scalar_t)(sample_num_w);
          scalar_t w1, w2, w3, w4, w5, w6, w7, w8;
          int x_low, x_high, y_low, y_high, z_low, z_high;

          bilinear_interpolate_gradient_3d<scalar_t>(
              depth, height, width, z, y, x, w1, w2, w3, w4, w5, w6, w7, w8, x_low, x_high, y_low, y_high, z_low, z_high);
          scalar_t g1 = offset_top_diff * w1 / count;
          scalar_t g2 = offset_top_diff * w2 / count;
          scalar_t g3 = offset_top_diff * w3 / count;
          scalar_t g4 = offset_top_diff * w4 / count;

          scalar_t g5 = offset_top_diff * w5 / count;
          scalar_t g6 = offset_top_diff * w6 / count;
          scalar_t g7 = offset_top_diff * w7 / count;
          scalar_t g8 = offset_top_diff * w8 / count;
                          
          if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0 && z_low >= 0 && z_high >= 0) {
            atomicAdd(offset_bottom_diff + (z_low * height + y_low) * width + x_low, g1);    // top_left_front
            atomicAdd(offset_bottom_diff + (z_low * height + y_low) * width + x_high, g2);   // top_right_front
            atomicAdd(offset_bottom_diff + (z_low * height + y_high) * width + x_low, g3);   // bottom_left_front  
            atomicAdd(offset_bottom_diff + (z_low * height + y_high) * width + x_high, g4);  // bottom_right_front

            atomicAdd(offset_bottom_diff + (z_high * height + y_low) * width + x_low, g5);   // top_left_back
            atomicAdd(offset_bottom_diff + (z_high * height + y_low) * width + x_high, g6);  // top_right_back
            atomicAdd(offset_bottom_diff + (z_high * height + y_high) * width + x_low, g7);  // bottom_left_back
            atomicAdd(offset_bottom_diff + (z_high * height + y_high) * width + x_high, g8); // bottom_right_back

            // previous implementation (incorrect)
            // atomicAdd(offset_bottom_diff + (y_low * width + x_low) * depth + z_low, g1);    // top_left_front
            // atomicAdd(offset_bottom_diff + (y_low * width + x_high) * depth + z_low, g2);   // top_right_front
            // atomicAdd(offset_bottom_diff + (y_high * width + x_low) * depth + z_low, g3);   // bottom_left_front  
            // atomicAdd(offset_bottom_diff + (y_high * width + x_high) * depth + z_low, g4);  // bottom_right_front

            // atomicAdd(offset_bottom_diff + (y_low * width + x_low) * depth + z_high, g5);   // top_left_back
            // atomicAdd(offset_bottom_diff + (y_low * width + x_high) * depth + z_high, g6);  // top_right_back
            // atomicAdd(offset_bottom_diff + (y_high * width + x_low) * depth + z_high, g7);  // bottom_left_back
            // atomicAdd(offset_bottom_diff + (y_high * width + x_high) * depth + z_high, g8); // bottom_right_back
          }
        }
      }
    }
  }
}

int ROIAlignBackwardLaucher(const at::Tensor top_grad, const at::Tensor rois,
                            const float spatial_scale, const int sample_num,
                            const int channels, const int height,
                            const int width, const int num_rois,
                            const int pooled_height, const int pooled_width,
                            at::Tensor bottom_grad) {
  const int output_size = num_rois * pooled_height * pooled_width * channels;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.type(), "ROIAlignLaucherBackward", ([&] {
        const scalar_t *top_diff = top_grad.data<scalar_t>();
        const scalar_t *rois_data = rois.data<scalar_t>();
        scalar_t *bottom_diff = bottom_grad.data<scalar_t>();
        if (sizeof(scalar_t) == sizeof(double)) {
          fprintf(stderr, "double is not supported\n");
          exit(-1);
        }

        ROIAlignBackward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, top_diff, rois_data, spatial_scale, sample_num,
                channels, height, width, pooled_height, pooled_width,
                bottom_diff);
      }));
  THCudaCheck(cudaGetLastError());
  return 1;
}

int ROIAlignBackwardLaucher3D(const at::Tensor top_grad, const at::Tensor rois,
                              const float spatial_scale, const float spatial_scale_depth, const int sample_num,
                              const int channels, const int depth, const int height,
                              const int width, const int num_rois,
                              const int pooled_depth, const int pooled_height, const int pooled_width,
                              at::Tensor bottom_grad) {
  const int output_size = num_rois * pooled_depth * pooled_height * pooled_width * channels;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.type(), "ROIAlignLaucherBackward3D", ([&] {
        const scalar_t *top_diff = top_grad.data<scalar_t>();
        const scalar_t *rois_data = rois.data<scalar_t>();
        scalar_t *bottom_diff = bottom_grad.data<scalar_t>();
        if (sizeof(scalar_t) == sizeof(double)) {
          fprintf(stderr, "double is not supported\n");
          exit(-1);
        }

        ROIAlignBackward3D<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, top_diff, rois_data, spatial_scale, spatial_scale_depth, sample_num,
                channels, depth, height, width, pooled_depth, pooled_height, pooled_width,
                bottom_diff);
      }));
  THCudaCheck(cudaGetLastError());
  return 1;
}