import mmcv
import numpy as np
import torch
from skimage.transform import resize
import matplotlib.pyplot as plt

__all__ = [
    'ImageTransform', 'BboxTransform', 'MaskTransform', 'SegMapTransform',
    'Numpy2Tensor'
]


class ImageTransform(object):
    """Preprocess an image.

    1. rescale the image to expected size
    2. normalize the image
    3. flip the image (if needed)
    4. pad the image (if needed)
    5. transpose to (c, h, w)
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 to_rgb=True,
                 size_divisor=None):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        self.size_divisor = size_divisor

    def __call__(self, img, scale, flip=False, keep_ratio=True):
        if keep_ratio:
            img, scale_factor = mmcv.imrescale(img, scale, return_scale=True)
        else:
            img, w_scale, h_scale = mmcv.imresize(
                img, scale, return_scale=True)
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
        img_shape = img.shape
        img = mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
        if flip:
            img = mmcv.imflip(img)
        if self.size_divisor is not None:
            img = mmcv.impad_to_multiple(img, self.size_divisor)
            pad_shape = img.shape
        else:
            pad_shape = img_shape
        img = img.transpose(2, 0, 1)
        return img, img_shape, pad_shape, scale_factor


def bbox_flip(bboxes, img_shape):
    """Flip bboxes horizontally.

    Args:
        bboxes(ndarray): shape (..., 4*k)
        img_shape(tuple): (height, width)
    """
    assert bboxes.shape[-1] % 4 == 0
    w = img_shape[1]
    flipped = bboxes.copy()
    flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
    flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
    return flipped


class BboxTransform(object):
    """Preprocess gt bboxes.

    1. rescale bboxes according to image size
    2. flip bboxes (if needed)
    3. pad the first dimension to `max_num_gts`
    """

    def __init__(self, max_num_gts=None):
        self.max_num_gts = max_num_gts

    def __call__(self, bboxes, img_shape, scale_factor, flip=False):
        if bboxes.shape[1] == 6:
            gt_bboxes = bboxes * scale_factor
            if flip:
                # TODO: need to reimplment flip for it to work with resized depth
                xy_gt_bboxes = bbox_flip(xy_gt_bboxes, img_shape)
            gt_bboxes[:, 0] = np.clip(gt_bboxes[:, 0], 0, img_shape[1] - 1)
            gt_bboxes[:, 2] = np.clip(gt_bboxes[:, 2], 0, img_shape[1] - 1)

            gt_bboxes[:, 1] = np.clip(gt_bboxes[:, 1], 0, img_shape[0] - 1)
            gt_bboxes[:, 3] = np.clip(gt_bboxes[:, 3], 0, img_shape[0] - 1)

            gt_bboxes[:, 4] = np.clip(gt_bboxes[:, 4], 0, img_shape[3] - 1)
            gt_bboxes[:, 5] = np.clip(gt_bboxes[:, 5], 0, img_shape[3] - 1)

            return gt_bboxes
        else:
            gt_bboxes = bboxes * scale_factor
            if flip:
                gt_bboxes = bbox_flip(gt_bboxes, img_shape)
            gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1] - 1)
            gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0] - 1)
            if self.max_num_gts is None:
                return gt_bboxes
            else:
                num_gts = gt_bboxes.shape[0]
                padded_bboxes = np.zeros((self.max_num_gts, 4), dtype=np.float32)
                padded_bboxes[:num_gts, :] = gt_bboxes
                return padded_bboxes


class MaskTransform(object):
    """Preprocess masks.

    1. resize masks to expected size and stack to a single array
    2. flip the masks (if needed)
    3. pad the masks (if needed)
    """

    def __call__(self, masks, pad_shape, scale_factor, flip=False, is3D=False):
        if is3D:
            new_masks = []
            for mask in masks:
                # mimic how cv2.resize is implemented: https://stackoverflow.com/questions/12569452/how-to-identify-numpy-types-in-python
                if scale_factor > 1.0:
                    mask = (resize(mask, (int(mask.shape[0] * scale_factor), int(mask.shape[1] * scale_factor), int(mask.shape[2] * scale_factor))) * 255).astype(np.uint8)
                mask[mask > 0] = 1
                new_masks.append(mask)
                # self.plt_mask_with_resized_mask(masks, new_masks, 1); ForkedPdb().set_trace() # debug only
            if flip:
                breakpoint() # TODO: flip not yet implemented
            padded_masks = np.stack(new_masks, axis=0)  
        else:
            masks = [
                mmcv.imrescale(mask, scale_factor, interpolation='nearest')
                for mask in masks
            ]
            if flip:
                masks = [mask[:, ::-1] for mask in masks]
            padded_masks = [
                mmcv.impad(mask, pad_shape[:2], pad_val=0) for mask in masks
            ]
            padded_masks = np.stack(padded_masks, axis=0)        
        return padded_masks
    
    def plt_mask_with_resized_mask(self, mask, resized_mask, bbox_num):
        for cur_slice in range(mask.shape[2]):
            filename = 'tests/bbox_{}_mask_{}.png'.format(bbox_num, cur_slice)
            plt.figure()
            plt.imshow(mask[:,:,cur_slice])
            plt.savefig(filename)
            plt.close()
        for cur_slice in range(resized_mask.shape[2]):
            filename = 'tests/bbox_{}_resized_mask{}.png'.format(bbox_num, cur_slice)
            plt.figure()
            plt.imshow(resized_mask[:,:,cur_slice])
            plt.savefig(filename)
            plt.close()


class SegMapTransform(object):
    """Preprocess semantic segmentation maps.

    1. rescale the segmentation map to expected size
    3. flip the image (if needed)
    4. pad the image (if needed)
    """

    def __init__(self, size_divisor=None):
        self.size_divisor = size_divisor

    def __call__(self, img, scale, flip=False, keep_ratio=True):
        if keep_ratio:
            img = mmcv.imrescale(img, scale, interpolation='nearest')
        else:
            img = mmcv.imresize(img, scale, interpolation='nearest')
        if flip:
            img = mmcv.imflip(img)
        if self.size_divisor is not None:
            img = mmcv.impad_to_multiple(img, self.size_divisor)
        return img


class Numpy2Tensor(object):

    def __init__(self):
        pass

    def __call__(self, *args):
        if len(args) == 1:
            return torch.from_numpy(args[0])
        else:
            return tuple([torch.from_numpy(np.array(array)) for array in args])
