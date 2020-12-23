import torch
import numpy as np
import mmcv
from skimage.transform import resize
import matplotlib.pyplot as plt


def mask_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list,
                cfg):
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    mask_targets = map(mask_target_single, pos_proposals_list,
                       pos_assigned_gt_inds_list, gt_masks_list, cfg_list)
    mask_targets = torch.cat(list(mask_targets))
    return mask_targets


def mask_target_single(pos_proposals, pos_assigned_gt_inds, gt_masks, cfg):
    mask_size = cfg.mask_size
    num_pos = pos_proposals.size(0)
    mask_targets = []
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        for i in range(num_pos):
            bbox = proposals_np[i, :].astype(np.int32)
            gt_mask = gt_masks[pos_assigned_gt_inds[i]]

            if bbox.shape[0] == 4: 
                x1, y1, x2, y2 = bbox
                w = np.maximum(x2 - x1 + 1, 1)
                h = np.maximum(y2 - y1 + 1, 1)
                # mask is uint8 both before and after resizing
                target = mmcv.imresize(gt_mask[y1:y1 + h, x1:x1 + w],
                                    (mask_size, mask_size))
            elif bbox.shape[0] == 6:
                mask_size_depth = cfg.mask_size_depth
                x1, y1, x2, y2, z1, z2 = bbox
                w = np.maximum(x2 - x1 + 1, 1)
                h = np.maximum(y2 - y1 + 1, 1)
                d = np.maximum(z2 - z1 + 1, 1)
                target = gt_mask[z1:z1 + d, y1:y1 + h, x1:x1 + w].cpu().numpy()
                target = 255 * resize(target, (mask_size_depth, mask_size, mask_size))
                target = target.astype(np.uint8)
                target[target > 0] = 1
            mask_targets.append(target)
        mask_targets = torch.from_numpy(np.stack(mask_targets)).float().to(
            pos_proposals.device)
    else:
        mask_targets = pos_proposals.new_zeros((0, mask_size, mask_size))
    return mask_targets

def plt_mask_with_resized_mask(mask, resized_mask, bbox_num):
    
    for cur_slice in range(mask.shape[0]):
        filename = 'tests/bbox_{}_mask_{}.png'.format(bbox_num, cur_slice)
        plt.figure()
        plt.imshow(mask[cur_slice,:,:])
        plt.savefig(filename)
        plt.close()
    for cur_slice in range(resized_mask.shape[0]):
        filename = 'tests/bbox_{}_resized_mask{}.png'.format(bbox_num, cur_slice)
        plt.figure()
        plt.imshow(resized_mask[cur_slice,:,:])
        plt.savefig(filename)
        plt.close()