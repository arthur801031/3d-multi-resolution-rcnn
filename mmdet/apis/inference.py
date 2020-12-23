import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch

from mmdet.core import get_classes
from mmdet.datasets import to_tensor
from mmdet.datasets.transforms import ImageTransform

from PIL import Image
import cv2

def _prepare_data(img, img_transform, cfg, device):
    ori_shape = img.shape
    img, img_shape, pad_shape, scale_factor = img_transform(
        img,
        scale=cfg.data.test.img_scale,
        keep_ratio=cfg.data.test.get('resize_keep_ratio', True))
    img = to_tensor(img).to(device).unsqueeze(0)
    img_meta = [
        dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=False)
    ]
    return dict(img=[img], img_meta=[img_meta])

def _prepare_data_3d(img_np, img_transform, cfg, device):
    ori_shape = (img_np.shape[0], img_np.shape[1], 3)
    total_num_slices = img_np.shape[2]
    imgs = []
    for cur_slice in range(total_num_slices):
        img = img_np[:,:,cur_slice]
        img = Image.fromarray(img).convert('RGB')
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img, img_shape, pad_shape, scale_factor = img_transform(
            img,
            scale=cfg.data.test.img_scale,
            keep_ratio=cfg.data.test.get('resize_keep_ratio', True))
        imgs.append(img)

    imgs = to_tensor(np.array(imgs)).to(device).unsqueeze(0)
    img_meta = [
        dict(
            ori_shape=ori_shape,
            img_shape=(*img_shape, total_num_slices),
            pad_shape=(*pad_shape, total_num_slices),
            scale_factor=scale_factor,
            flip=False)
    ]
    imgs = imgs.permute(0, 2, 1, 3, 4)
    assert imgs.shape[1] == 3 # make sure channel size is 3
    return dict(imgs=imgs, img_meta=[img_meta])

def _prepare_data_3d_2scales(img_np, img_np_2, img_transform, cfg, device):
    ori_shape = (img_np.shape[0], img_np.shape[1], 3)
    ori_shape_2 = (img_np_2.shape[0], img_np_2.shape[1], 3)
    total_num_slices = img_np.shape[2]
    total_num_slices_2 = img_np_2.shape[2]

    # first image
    imgs = []
    for cur_slice in range(total_num_slices):
        img = img_np[:,:,cur_slice]
        img = Image.fromarray(img).convert('RGB')
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img, img_shape, pad_shape, scale_factor = img_transform(
            img,
            scale=cfg.data.test.img_scale,
            keep_ratio=cfg.data.test.get('resize_keep_ratio', True))
        imgs.append(img)

    imgs = to_tensor(np.array(imgs)).to(device).unsqueeze(0)
    img_meta = [
        dict(
            ori_shape=ori_shape,
            img_shape=(*img_shape, total_num_slices),
            pad_shape=(*pad_shape, total_num_slices),
            # scale_factor=1.0 / (img_np_2.shape[0] / img_np.shape[0]), # scale up to 1.5x
            scale_factor=1.0, # scale down 1.0x
            flip=False)
    ]
    imgs = imgs.permute(0, 2, 1, 3, 4)

    # second image
    imgs_2 = []
    for cur_slice in range(total_num_slices_2):
        img = img_np_2[:,:,cur_slice]
        img = Image.fromarray(img).convert('RGB')
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img, img_shape, pad_shape, scale_factor = img_transform(
            img,
            scale=cfg.data2_2scales.test.img_scale,
            keep_ratio=cfg.data.test.get('resize_keep_ratio', True))
        imgs_2.append(img)

    imgs_2 = to_tensor(np.array(imgs_2)).to(device).unsqueeze(0)
    img_meta_2 = [
        dict(
            ori_shape=ori_shape_2,
            img_shape=(*img_shape, total_num_slices_2),
            pad_shape=(*pad_shape, total_num_slices_2),
            # scale_factor=scale_factor, # scale up to 1.5x
            scale_factor=1.5, # scale down 1.0x
            flip=False)
    ]
    imgs_2 = imgs_2.permute(0, 2, 1, 3, 4)

    assert imgs.shape[1] == 3 # make sure channel size is 3
    assert imgs_2.shape[1] == 3
    return dict(imgs=imgs, img_meta=[img_meta], imgs_2=imgs_2, img_meta_2=[img_meta_2])

def _inference_single(model, img, img_transform, cfg, device):
    img = mmcv.imread(img)
    data = _prepare_data(img, img_transform, cfg, device)
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result

def _inference_single_3d(model, img, img_transform, cfg, device):
    img_np = np.load(img)
    data = _prepare_data_3d(img_np, img_transform, cfg, device)
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result

def _inference_single_3d_2scales(model, img, img_2, img_transform, cfg, device):
    img_np = np.load(img)
    img_np_2 = np.load(img_2)
    data = _prepare_data_3d_2scales(img_np, img_np_2, img_transform, cfg, device)
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result

def _inference_generator(model, imgs, img_transform, cfg, device):
    for img in imgs:
        yield _inference_single(model, img, img_transform, cfg, device)

def _inference_generator_3d(model, imgs, img_transform, cfg, device):
    for img in imgs:
        yield _inference_single_3d(model, img, img_transform, cfg, device)

def _inference_generator_3d_2scales(model, imgs, imgs_2, img_transform, cfg, device):
    for img, img_2 in zip(imgs, imgs_2):
        assert img.split('/')[-1] == img_2.split('/')[-1]
        yield _inference_single_3d_2scales(model, img, img_2, img_transform, cfg, device)

def inference_detector(model, imgs, cfg, device='cuda:0'):
    img_transform = ImageTransform(
        size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)
    model = model.to(device)
    model.eval()

    if not isinstance(imgs, list):
        return _inference_single(model, imgs, img_transform, cfg, device)
    else:
        return _inference_generator(model, imgs, img_transform, cfg, device)

def inference_detector_3d(model, imgs, cfg, device='cuda:0'):
    img_transform = ImageTransform(
        size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)
    model = model.to(device)
    model.eval()

    if not isinstance(imgs, list):
        return _inference_single_3d(model, imgs, img_transform, cfg, device)
    else:
        return _inference_generator_3d(model, imgs, img_transform, cfg, device)

def inference_detector_3d_2scales(model, imgs, imgs_2, cfg, device='cuda:0'):
    img_transform = ImageTransform(
        size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)
    model = model.to(device)
    model.eval()

    if not isinstance(imgs, list):
        return _inference_single_3d_2scales(model, imgs, imgs_2, img_transform, cfg, device)
    else:
        return _inference_generator_3d_2scales(model, imgs, imgs_2, img_transform, cfg, device)

def show_result(img, result, dataset='coco', score_thr=0.3, out_file=None, font_scale=0.5):
    img = mmcv.imread(img)
    class_names = get_classes(dataset)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(
                0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)

    write_bboxes_to_npy(bboxes, out_file)

    mmcv.imshow_det_bboxes(
        img.copy(),
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,
        show=out_file is None,
        out_file=out_file,
        font_scale=font_scale)

def show_result_3d(img, result, dataset='coco', score_thr=0.3, out_file=None, font_scale=0.5):
    img_np = np.load(img)    
    class_names = get_classes(dataset)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(
                0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5

    bboxes_placeholders = [[] for i in range(0, 160)]
    for bbox in bboxes:
        for z_index in range(int(np.floor(bbox[4])), int(np.ceil(bbox[5])+ 1)):
            bboxes_placeholders[z_index].append([bbox[0], bbox[1], bbox[2], bbox[3], bbox[6]])
    
    for index, boxes in enumerate(bboxes_placeholders):
        if len(boxes) > 0:
            img = img_np[:,:,index]
            img = Image.fromarray(img).convert('RGB')
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            labels = np.array([0 for i in range(len(boxes))])

            mmcv.imshow_det_bboxes(
            img.copy(),
            np.array(boxes),
            labels,
            class_names=class_names,
            score_thr=score_thr,
            show=out_file is None,
            out_file=out_file.split('.')[-2] + '-{}.png'.format(index),
            font_scale=0)


def display_result_3d(img, result, dataset='coco', score_thr=0.3):
    img_np = np.load(img)    
    class_names = get_classes(dataset)
    bbox_result = result
    bboxes = np.vstack(bbox_result)
   
    bboxes_placeholders = [[] for i in range(0, 160)]
    for bbox in bboxes:
        for z_index in range(int(np.floor(bbox[4])), int(np.ceil(bbox[5])+ 1)):
            bboxes_placeholders[z_index].append([bbox[0], bbox[1], bbox[2], bbox[3], bbox[6]])
    
    for index, boxes in enumerate(bboxes_placeholders):
        if len(boxes) > 0:
            for box in boxes:
                if box[4] > score_thr:
                    print('slice {} score {}'.format(index, box[4]))

'''
write bounding boxes result to npy file
'''
def write_bboxes_to_npy(bboxes, out_file):
    if out_file is not None:
        bboxes_filename = out_file.split('.')[0]            # A001-2342.jpeg => A001-2342
        bboxes_filename = bboxes_filename + '.npy'
        np.save(bboxes_filename, bboxes)