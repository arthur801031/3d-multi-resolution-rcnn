import torch
import cv2
import numpy as np

from ..bbox import assign_and_sample, build_assigner, PseudoSampler, bbox2delta, bbox2delta3d
from ..utils import multi_apply


def anchor_target(anchor_list,
                  valid_flag_list,
                  gt_bboxes_list,
                  img_metas,
                  target_means,
                  target_stds,
                  cfg,
                  gt_bboxes_ignore_list=None,
                  gt_labels_list=None,
                  label_channels=1,
                  sampling=True,
                  unmap_outputs=True,
                  iteration=None):
    """Compute regression and classification targets for anchors.

    Args:
        anchor_list (list[list]): Multi level anchors of each image.
        valid_flag_list (list[list]): Multi level valid flags of each image.
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
        img_metas (list[dict]): Meta info of each image.
        target_means (Iterable): Mean value of regression targets.
        target_stds (Iterable): Std value of regression targets.
        cfg (dict): RPN train configs.

    Returns:
        tuple
    """
    num_imgs = len(img_metas)
    assert len(anchor_list) == len(valid_flag_list) == num_imgs

    # anchor number of multi levels
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    # concat all level anchors and flags to a single tensor
    for i in range(num_imgs):
        assert len(anchor_list[i]) == len(valid_flag_list[i])
        anchor_list[i] = torch.cat(anchor_list[i])
        valid_flag_list[i] = torch.cat(valid_flag_list[i])

    # compute targets for each image
    if gt_bboxes_ignore_list is None:
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
    if gt_labels_list is None:
        gt_labels_list = [None for _ in range(num_imgs)]
    (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
     pos_inds_list, neg_inds_list, inside_flags) = multi_apply(
         anchor_target_single,
         anchor_list,
         valid_flag_list,
         gt_bboxes_list,
         gt_bboxes_ignore_list,
         gt_labels_list,
         img_metas,
         target_means=target_means,
         target_stds=target_stds,
         cfg=cfg,
         label_channels=label_channels,
         sampling=sampling,
         unmap_outputs=unmap_outputs,
         iteration=iteration)
    # no valid anchors
    if any([labels is None for labels in all_labels]):
        return None
    # sampled anchors of all images
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
    # split targets to a list w.r.t. multiple levels
    labels_list = images_to_levels(all_labels, num_level_anchors)
    label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
    bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
    bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
    anchors_list = images_to_levels(anchor_list, num_level_anchors)
    inside_flags = images_to_levels(inside_flags, num_level_anchors)
    return (labels_list, label_weights_list, bbox_targets_list,
            bbox_weights_list, num_total_pos, num_total_neg, anchors_list, inside_flags)

def get_anchor_inside_flags(anchor_list, valid_flag_list, img_metas):
    # anchor number of multi levels
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

    num_imgs = len(img_metas)
    # concat all level anchors and flags to a single tensor
    for i in range(num_imgs):
        assert len(anchor_list[i]) == len(valid_flag_list[i])
        anchor_list[i] = torch.cat(anchor_list[i])
        valid_flag_list[i] = torch.cat(valid_flag_list[i])

    (_, inside_flags) = multi_apply(
         get_anchor_inside_flags_single,
         anchor_list,
         valid_flag_list,
         img_metas)
    inside_flags = images_to_levels(inside_flags, num_level_anchors)
    return inside_flags
    

def get_anchor_inside_flags_single(flat_anchors, valid_flags, img_meta):
    inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                       img_meta['img_shape'], 0)
    return (0, inside_flags) # return 0 because returning 1 item for multi_apply causes problem 


def images_to_levels(target, num_level_anchors):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_targets.append(target[:, start:end].squeeze(0))
        start = end
    return level_targets


def anchor_target_single(flat_anchors,
                         valid_flags,
                         gt_bboxes,
                         gt_bboxes_ignore,
                         gt_labels,
                         img_meta,
                         target_means,
                         target_stds,
                         cfg,
                         label_channels=1,
                         sampling=True,
                         unmap_outputs=True,
                         iteration=None):
    inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                       img_meta['img_shape'],
                                       cfg.allowed_border)
    if not inside_flags.any():
        return (None, ) * 6
    # assign gt and sample anchors
    anchors = flat_anchors[inside_flags, :]

    if sampling:
        assign_result, sampling_result = assign_and_sample(
            anchors, gt_bboxes, gt_bboxes_ignore, None, cfg)
    else:
        bbox_assigner = build_assigner(cfg.assigner)
        assign_result = bbox_assigner.assign(anchors, gt_bboxes,
                                             gt_bboxes_ignore, gt_labels)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchors,
                                              gt_bboxes)

    num_valid_anchors = anchors.shape[0]
    bbox_targets = torch.zeros_like(anchors)
    bbox_weights = torch.zeros_like(anchors)
    labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
    label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    
    # debug only: check whether or not anchors and gt_bboxes actually overlap
    # show_bboxes_gt_bboxes(sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes, iteration=iteration)

    if len(pos_inds) > 0:
        if gt_bboxes.shape[1] == 6:
            pos_bbox_targets = bbox2delta3d(sampling_result.pos_bboxes,
                                        sampling_result.pos_gt_bboxes,
                                        target_means, target_stds)
        elif gt_bboxes.shape[1] == 4:
            pos_bbox_targets = bbox2delta(sampling_result.pos_bboxes,
                                        sampling_result.pos_gt_bboxes,
                                        target_means, target_stds)
        bbox_targets[pos_inds, :] = pos_bbox_targets
        bbox_weights[pos_inds, :] = 1.0
        if gt_labels is None:
            labels[pos_inds] = 1
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0

    # map up to original set of anchors
    if unmap_outputs:
        num_total_anchors = flat_anchors.size(0)
        labels = unmap(labels, num_total_anchors, inside_flags)
        label_weights = unmap(label_weights, num_total_anchors, inside_flags)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

    return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
            neg_inds, inside_flags)


def anchor_inside_flags(flat_anchors, valid_flags, img_shape,
                        allowed_border=0):
    img_h, img_w = img_shape[:2]
    if flat_anchors.shape[1] == 6:
        img_d = img_shape[3]
        if allowed_border >= 0:
            inside_flags = valid_flags & \
                (flat_anchors[:, 0] >= -allowed_border) & \
                (flat_anchors[:, 1] >= -allowed_border) & \
                (flat_anchors[:, 4] >= -allowed_border) & \
                (flat_anchors[:, 2] < img_w + allowed_border) & \
                (flat_anchors[:, 3] < img_h + allowed_border) & \
                (flat_anchors[:, 5] < img_d + allowed_border)
        else:
            inside_flags = valid_flags
    elif flat_anchors.shape[1] == 4:
        if allowed_border >= 0:
            inside_flags = valid_flags & \
                (flat_anchors[:, 0] >= -allowed_border) & \
                (flat_anchors[:, 1] >= -allowed_border) & \
                (flat_anchors[:, 2] < img_w + allowed_border) & \
                (flat_anchors[:, 3] < img_h + allowed_border)
        else:
            inside_flags = valid_flags
    
    return inside_flags


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

def imwrite(img, file_path, params=None):
    """Write image to file
    """
    return cv2.imwrite(file_path, img, params)

def show_bboxes_gt_bboxes(bboxes,
                    gt_bboxes=None,
                    colors=[(0, 255, 0), (0, 0, 255)],
                    top_k=-1,
                    thickness=1,
                    show=True,
                    iteration=None):
        """Draw bboxes on an image.
        """
        assert bboxes.shape == gt_bboxes.shape
        bboxes = bboxes.cpu().numpy()
        bboxes = [bboxes]
        gt_bboxes = gt_bboxes.cpu().numpy()
        gt_bboxes = [gt_bboxes]
        # keys = {'xy', 'xz', 'yz'} 
        keys = {'xy'} 

        for key in keys:
            if key == 'xy':
                img = np.zeros((512, 512, 3))
            elif key == 'xz':
                img = np.zeros((160, 512, 3))
            elif key == 'yz':
                img = np.zeros((160, 512, 3))

            if len(bboxes) > 0: 
                for i, _bboxes in enumerate(bboxes):
                    _bboxes = _bboxes.astype(np.int32)
                    if top_k <= 0:
                        _top_k = _bboxes.shape[0]
                    else:
                        _top_k = min(top_k, _bboxes.shape[0])
                    for j in range(_top_k):
                        if key == 'xy':
                            left_top = (_bboxes[j, 0], _bboxes[j, 1])
                            right_bottom = (_bboxes[j, 2], _bboxes[j, 3])
                        elif key == 'xz':
                            left_top = (_bboxes[j, 0], _bboxes[j, 4])
                            right_bottom = (_bboxes[j, 2], _bboxes[j, 5])
                        elif key == 'yz':
                            left_top = (_bboxes[j, 1], _bboxes[j, 4])
                            right_bottom = (_bboxes[j, 3], _bboxes[j, 5])

                        cv2.rectangle(img, left_top, right_bottom, colors[0], thickness=thickness)
            
            if gt_bboxes is not None and len(gt_bboxes) > 0:
                # gt_bboxes
                for i, _gt_bboxes in enumerate(gt_bboxes):
                    _gt_bboxes = _gt_bboxes.astype(np.int32)
                    if top_k <= 0:
                        _top_k = _gt_bboxes.shape[0]
                    else:
                        _top_k = min(top_k, _gt_bboxes.shape[0])
                    for j in range(_top_k):
                        if key == 'xy':
                            left_top = (_gt_bboxes[j, 0], _gt_bboxes[j, 1])
                            right_bottom = (_gt_bboxes[j, 2], _gt_bboxes[j, 3])
                        elif key == 'xz':
                            left_top = (_gt_bboxes[j, 0], _gt_bboxes[j, 4])
                            right_bottom = (_gt_bboxes[j, 2], _gt_bboxes[j, 5])
                        elif key == 'yz':
                            left_top = (_gt_bboxes[j, 1], _gt_bboxes[j, 4])
                            right_bottom = (_gt_bboxes[j, 3], _gt_bboxes[j, 5])
                        cv2.rectangle( img, left_top, right_bottom, colors[1], thickness=1)
            imwrite(img, 'tests2/iter_{}_{}.png'.format(iteration, key))