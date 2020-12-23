import mmcv
import numpy as np
from numpy import random

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps, bbox_overlaps_3d


class PhotoMetricDistortion(object):

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, img, boxes, labels, mask=None):
        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]

        return img, boxes, labels


class Expand(object):

    def __init__(self, mean=(0, 0, 0), to_rgb=True, ratio_range=(1, 4)):
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range

    def __call__(self, img, boxes, labels, mask=None):
        if random.randint(2):
            return img, boxes, labels

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean).astype(img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img
        img = expand_img
        boxes += np.tile((left, top), 2)
        return img, boxes, labels


class RandomCrop(object):

    def __init__(self,
                 min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                 min_crop_size=0.3):
        # 1: return ori img
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size

    def __call__(self, img, boxes, labels, mask=None):
        h, w, c = img.shape
        while True:
            mode = random.choice(self.sample_mode)
            if mode == 1:
                return img, boxes, labels

            min_iou = mode
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array((int(left), int(top), int(left + new_w),
                                  int(top + new_h)))
                overlaps = bbox_overlaps(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = (center[:, 0] > patch[0]) * (
                    center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * (
                        center[:, 1] < patch[3])
                if not mask.any():
                    continue
                boxes = boxes[mask]
                labels = labels[mask]

                # adjust boxes
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                boxes -= np.tile(patch[:2], 2)

                return img, boxes, labels


class RandomCrop3D(object):

    def __init__(self,
                 min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                 min_crop_size=0.3):
        # 1: return ori img
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size

    def retrieve_valid_coordinates(self, img_dim, patch_dim, boxes_left, boxes_right):
        mask = None
        while mask is None or not mask.any():
            corner = random.randint(img_dim - patch_dim)
            mask = (boxes_left >= int(corner)) * (boxes_right < int(corner + patch_dim))
        return corner

    def __call__(self, img_np, boxes, labels, mask_np=None):
        h, w, d = img_np.shape
        
        # fixed patch width, height, and depth: should be the same as validation's image shape
        new_w = int(w / 4)
        new_h = int(h / 4)
        new_d = d

        while True:
            mode = random.choice(self.sample_mode)
            min_iou = mode

            left = self.retrieve_valid_coordinates(w, new_w, boxes[:, 0], boxes[:, 2])
            top = self.retrieve_valid_coordinates(h, new_h, boxes[:, 1], boxes[:, 3])
            front = 0

            patch = np.array((int(left), int(top), int(left + new_w),
                            int(top + new_h), int(front), int(front + new_d)))

            overlaps = bbox_overlaps_3d(
                    patch.reshape(-1, 6), boxes.reshape(-1, 6)).reshape(-1)
            if overlaps.min() < min_iou:
                continue

            # bounding boxes must be within the crop region (not touching the edges)
            mask = (boxes[:, 0] >= patch[0]) * (boxes[:, 1] >= patch[1]) * (boxes[:, 4] >= patch[4]) * \
                    (boxes[:, 2] < patch[2]) * (boxes[:, 3] < patch[3]) * (boxes[:, 5] < patch[5]) 
            if not mask.any():
                continue

            final_boxes = boxes[mask]
            final_labels = labels[mask]
            if len(final_boxes) == 0:
                ForkedPdb().set_trace() # should never happen
        
            if mask_np is not None:
                new_mask_np, index = [], 0
                for cur_mask in mask_np:
                    if mask[index]:
                        new_mask_np.append(cur_mask[patch[1]:patch[3], patch[0]:patch[2], patch[4]:patch[5]])
                    index += 1
                new_mask_np = np.array(new_mask_np)
            else:
                new_mask_np = None

            # adjust boxes
            img_np = img_np[patch[1]:patch[3], patch[0]:patch[2], patch[4]:patch[5]]
            final_boxes -= np.array((patch[0], patch[1], patch[0], patch[1], patch[4], patch[4]))

            return img_np, final_boxes, final_labels, new_mask_np



class ExtraAugmentation(object):

    def __init__(self,
                 photo_metric_distortion=None,
                 expand=None,
                 random_crop=None,
                 random_crop_3d=None):
        self.transforms = []
        if photo_metric_distortion is not None:
            self.transforms.append(
                PhotoMetricDistortion(**photo_metric_distortion))
        if expand is not None:
            self.transforms.append(Expand(**expand))
        if random_crop is not None:
            self.transforms.append(RandomCrop(**random_crop))
        if random_crop_3d is not None:
            self.transforms.append(RandomCrop3D(**random_crop_3d))

    def __call__(self, img, boxes, labels, mask=None):
        img = img.astype(np.float32)
        for transform in self.transforms:
            img, boxes, labels, mask = transform(img, boxes, labels, mask)
        return img, boxes, labels, mask
