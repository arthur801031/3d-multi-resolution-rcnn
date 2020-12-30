import mmcv
import cv2
import os
import json
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector_3d, show_result_3d, display_result_3d, inference_detector_3d_2scales
from mmdet.core.evaluation.coco_utils import nms_3d_python, overlap_in_precomputed_proposals_inference_mode
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps_3d
from imageio import imread
from skimage import measure, color
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pts
from pycocotools_local.coco import *
from pycocotools_local.cocoeval import *
from tqdm import tqdm
import time
import shutil
import pickle

def get_files_paths(direc):
    filenames = sorted([f for f in os.listdir(direc)])
    return ['{}/{}'.format(direc, filename) for filename in filenames]

def get_similar_file_path(direc, partial_name):
    for f in os.listdir(direc):
        if partial_name in f:
            return f
    return None

def create_or_empty_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

coco_dir = 'data/Stroke_v4-14TB/COCO-Test-Patches-WithNoCMB'
out_valid_annotate_dir = '{}/3d_mutli_resolution_rcnn'.format(coco_dir)
valid_subset_dir = '{}/test/COCO_test2019'.format(coco_dir)
valid_subset_full_dir = '{}/test/COCO_test_full2019'.format(coco_dir)
valid_anno_full_dir = '{}/test/annotations_full'.format(coco_dir)
instance_valid_path = '{}/annotations/instances_test2019.json'.format(coco_dir)
full_gt_annotations_path = '{}/annotations/instances_test2019_full.json'.format(coco_dir)
score_thr = 0.2
nms_score_thr = 0.1

coco_dir_2 = 'data/Stroke_v4-14TB/COCO-Test-Patches-WithNoCMB-1dot5x'
valid_subset_full_dir_2 = '{}/test/COCO_test_full2019'.format(coco_dir_2)
full_gt_annotations_path_2 = '{}/annotations/instances_test2019_full.json'.format(coco_dir_2)
valid_anno_full_dir_2 = '{}/test/annotations_full'.format(coco_dir_2)

mask_width = 512
mask_height = 512
mask_depth = 160
max_full_vol_slices = 160

cfg = mmcv.Config.fromfile('configs/3d-multi-resolution-rcnn.py')
cfg.model.pretrained = None

# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
_ = load_checkpoint(model, 'work_dirs/checkpoints/3d-multi-resolution-rcnn/latest.pth')

files_paths = get_files_paths(valid_subset_full_dir)
files_paths_2 = get_files_paths(valid_subset_full_dir_2)
files_anno_paths = get_files_paths(valid_anno_full_dir)
files_anno_paths_2 = get_files_paths(valid_anno_full_dir_2)
create_or_empty_folder(out_valid_annotate_dir)

coco_full_gt = COCO(full_gt_annotations_path) # for 512x512x160
full_filename_to_id = dict()
for img_id in coco_full_gt.getImgIds():
    full_filename_to_id[coco_full_gt.loadImgs([img_id])[0]['file_name']] = img_id

pbar, pCounter = tqdm(total=len(files_paths)), 1
for i, result in enumerate(inference_detector_3d_2scales(model, files_paths, files_paths_2, cfg, device='cuda:0')):
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None

    predicted_bboxes_placeholders = [[] for i in range(0, max_full_vol_slices)]
    gt_bboxes_placeholders = [[] for i in range(0, max_full_vol_slices)]
    
    filename = files_paths[i].split('/')[-1]
    img_id = full_filename_to_id[filename]
    annotation_ids = coco_full_gt.getAnnIds(imgIds=[img_id], iscrowd=None)
    annotations = coco_full_gt.loadAnns(annotation_ids)
    
    for ann in annotations:
        bbox = ann['bbox']
        for z_index in range(bbox[4], bbox[4]+bbox[5]):
            gt_bboxes_placeholders[z_index].append([bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]])
    if segm_result is None:
        for bbox in bbox_result[0]:
            for z_index in range(int(round(bbox[4])), int(round(bbox[5]))):
                predicted_bboxes_placeholders[z_index].append([bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1], bbox[4], bbox[5]-bbox[4], bbox[6]])
    else:
        predicted_masks_placeholders = [[] for i in range(0, max_full_vol_slices)]
        for bbox, segm in zip(bbox_result[0], segm_result[0]):            
            for z_index in range(int(round(bbox[4])), int(round(bbox[5]))):
                predicted_bboxes_placeholders[z_index].append([bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1], bbox[4], bbox[5]-bbox[4], bbox[6]])
                predicted_masks_placeholders[z_index].append(segm[z_index,:,:])
        
    full_volume = valid_subset_full_dir + '/' + filename # for 512x512x160
    full_volume = np.load(full_volume)
    index = 0
    if segm_result is None:
        for pred_bboxes, gt_bboxes in zip(predicted_bboxes_placeholders, gt_bboxes_placeholders):
            if len(pred_bboxes) > 0 or len(gt_bboxes) > 0:
                fig = plt.figure(figsize=(20, 20))
                plt.imshow(full_volume[:,:,index], cmap='gray'); plt.axis('off'); plt.title('{} slice {}'.format(filename, index))
                ax = plt.gca()
                if len(pred_bboxes) > 0:
                    for pred_bbox in pred_bboxes:
                        if pred_bbox[6] > score_thr:
                            rect = pts.Rectangle((pred_bbox[0], pred_bbox[1]), pred_bbox[2], pred_bbox[3], fill=False, edgecolor='red', linewidth=3, alpha=0.5)
                            ax.add_patch(rect)
                            if len(gt_bboxes) > 0:
                                gt_bboxes_reformat = []
                                for gt_bbox in gt_bboxes:
                                    gt_bboxes_reformat.append([gt_bbox[0], gt_bbox[1], gt_bbox[2]+gt_bbox[0], gt_bbox[3]+gt_bbox[1], gt_bbox[4], gt_bbox[5]+gt_bbox[4]])
                                pred_bbox_reformat = [[pred_bbox[0], pred_bbox[1], pred_bbox[2]+pred_bbox[0], pred_bbox[3]+pred_bbox[1], pred_bbox[4], pred_bbox[5]+pred_bbox[4]]]
                                
                                overlaps = bbox_overlaps_3d(np.array(gt_bboxes_reformat), np.array(pred_bbox_reformat))
                                target_iou = np.amax(overlaps)
                                if target_iou > 0:
                                    plt.text(pred_bbox[0]+15, pred_bbox[1]+15, "{0:.4f}".format(target_iou), fontsize='small', color='deepskyblue')
                                   
                if len(gt_bboxes) > 0:
                    for gt_bbox in gt_bboxes:
                        rect = pts.Rectangle((gt_bbox[0], gt_bbox[1]), gt_bbox[2], gt_bbox[3], fill=False, edgecolor='green', linewidth=3, alpha=0.5)
                        ax.add_patch(rect)
                fig.savefig('{}/{}_{}.png'.format(out_valid_annotate_dir, filename, index))
                plt.close()
            index += 1
    else:
        
        # load ground truth masks
        cur_annos = np.zeros((mask_width, mask_height, mask_depth))
        filename_nonpy = filename.split('.')[0]
        for file_anno_path in files_anno_paths: # for 512x512x160
            if filename_nonpy in file_anno_path:
                anno = np.load(file_anno_path)
                cur_annos += anno
        cur_annos[cur_annos>0] = 1
        gt_masks_placeholders = [[] for i in range(0, max_full_vol_slices)]
        for cur_slice in range(cur_annos.shape[2]):
            if len(np.unique(cur_annos[:,:,cur_slice])) > 1:
                gt_masks_placeholders[cur_slice].append(cur_annos[:,:,cur_slice])                
         
        for pred_bboxes, pred_masks, gt_bboxes, gt_mask in zip(predicted_bboxes_placeholders, predicted_masks_placeholders, gt_bboxes_placeholders, gt_masks_placeholders):
            if len(pred_bboxes) > 0 or len(gt_bboxes) > 0:
                fig = plt.figure(figsize=(20, 20))
                plt.imshow(full_volume[:,:,index], cmap='gray'); plt.axis('off'); plt.title('{} slice {}'.format(filename, index))
                ax = plt.gca()
                if len(pred_bboxes) > 0:
                    accum_masks = np.zeros((mask_width, mask_height))
                    for pred_bbox, pred_mask in zip(pred_bboxes, pred_masks):
                        if pred_bbox[4] > score_thr:
                            rect = pts.Rectangle((pred_bbox[0], pred_bbox[1]), pred_bbox[2], pred_bbox[3], fill=False, edgecolor='red', linewidth=3, alpha=0.5)
                            ax.add_patch(rect)
                            accum_masks += pred_mask
                    accum_masks[accum_masks > 0] = 1
                    # set background to transparent
                    accum_masks=accum_masks.astype(np.float) 
                    accum_masks[np.where(accum_masks==0)]=np.nan
                    plt.imshow(accum_masks, alpha=0.5, cmap='bwr', vmin=0, vmax=1)
                if len(gt_bboxes) > 0:
                    for gt_bbox in gt_bboxes:
                        rect = pts.Rectangle((gt_bbox[0], gt_bbox[1]), gt_bbox[2], gt_bbox[3], fill=False, edgecolor='green', linewidth=3, alpha=0.5)
                        ax.add_patch(rect)
                    gt_mask = gt_mask[0]
                    # set background to transparent
                    gt_mask=gt_mask.astype(np.float) 
                    gt_mask[np.where(gt_mask==0)]=np.nan
                    plt.imshow(gt_mask, alpha=0.2, cmap='winter', vmin=0, vmax=1)
                fig.savefig('{}/{}_{}.png'.format(out_valid_annotate_dir, filename, index))
                plt.close()
            index += 1
        
    pbar.update(pCounter)