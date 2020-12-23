from functools import partial

import mmcv
import numpy as np
from six.moves import map, zip


def tensor2imgs(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = mmcv.imdenormalize(
            img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs

def tensor2img3DNPrint(tensor, slice_num=85, bboxes=np.array([[0, 0, 0, 0]]), mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375), to_rgb=True):
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    # get the first image
    imgs = tensor[0, ...].cpu().numpy().transpose(2, 3, 1, 0)
    # get the first slice
    img = imgs[:,:,slice_num,:]
    img = mmcv.imdenormalize(img, mean, std, to_bgr=False)
    img = np.ascontiguousarray(img)
    mmcv.imshow_bboxes(img, bboxes)
    return img

def tensor2img3D(tensor, slice_num=85, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375), to_rgb=True):
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    # get the first image
    imgs = tensor[0, ...].cpu().numpy().transpose(2, 3, 1, 0)
    # get the first slice
    img = imgs[:,:,slice_num,:]
    img = mmcv.imdenormalize(img, mean, std, to_bgr=to_rgb).astype(np.uint8)
    img = np.ascontiguousarray(img)
    return img

def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret
