import mmcv
import numpy as np
from pycocotools_local.coco import *
from pycocotools_local.cocoeval import *
import pickle
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps_3d
from tqdm import tqdm

from .recall import eval_recalls


def coco_eval(result_file, result_types, coco, max_dets=(100, 300, 1000), slice_label=None, is3D=False, hasMask=False, full_filename_to_id=None, isParcellized=False):
    for res_type in result_types:
        assert res_type in [
            'proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'
        ]

    if mmcv.is_str(coco):
        coco = COCO(coco)
    assert isinstance(coco, COCO)

    if result_types == ['proposal_fast']:
        ar = fast_eval_recall(result_file, coco, np.array(max_dets))
        for i, num in enumerate(max_dets):
            print('AR@{}\t= {:.4f}'.format(num, ar[i]))
        return

    if hasMask:
        coco_dets = coco.loadRes3D(result_file)
    else:
        assert result_file.endswith('.json')
        coco_dets = coco.loadRes(result_file)

    img_ids = coco.getImgIds()
    for res_type in result_types:
        iou_type = 'bbox' if res_type == 'proposal' else res_type
        if iou_type == 'segm' and full_filename_to_id is not None:
            def evaluate_each_image_separately():
                final_results = dict()
                # aggregates results and perform summary for the combined results
                for filename, img_id in full_filename_to_id.items():
                    cocoEval = COCOeval(coco, coco_dets, iou_type, is3D=True)
                    cocoEval.params.imgIds = [img_id]
                    cocoEval.evaluate(target_id=img_id)
                    cocoEval.accumulate()
                    final_results[img_id] = cocoEval.eval['saved']
                    del cocoEval
                cocoEval = COCOeval(coco, coco_dets, iou_type, is3D=True)
                # save final_results in case cocoeval has a bug
                with open('./accum_masks.pkl', 'wb') as f:
                    pickle.dump(final_results, f, pickle.HIGHEST_PROTOCOL)
                # load saved masks.pkl if something bad happens during evaluation
                # with open('./accum_masks.pkl', 'rb') as f:
                #     final_results = pickle.load(f)
                cocoEval.accumulate_all(final_results)
                cocoEval.summarize()

            # evaluate each image separately due to loading all images' segmentation masks taking up too much space
            evaluate_each_image_separately()

            # original code
            # cocoEval = COCOeval(coco, coco_dets, iou_type, is3D=True)
            # cocoEval.params.imgIds = img_ids
            # cocoEval.evaluate(slice_label=slice_label)
            # cocoEval.accumulate()
            # cocoEval.summarize()

        else:
            if isParcellized:
                cocoEval = COCOeval(coco, coco_dets, iou_type, is3D=True, isParcellized=True)
            elif is3D:
                cocoEval = COCOeval(coco, coco_dets, iou_type, is3D=True)
            else:
                cocoEval = COCOeval(coco, coco_dets, iou_type)

            
            ############## output each patient's individual result for each brain region #############
            brain_regions_names = [
                'Caudate',
                'Putamen',
                'Pallidum',
                'Thalamus',
                'Corpus_Callosum',
                'Sub_Lobar',
                'Parietal',
                'Frontal',
                'Frontal_Temporal',
                'Limbic',
                'Occipital',
                'Temporal',
                'Cerebellum',
                'Brainstem',
            ]

            target_filenames = [
                'A001-23230277_instance_v1.npy',
                'A012-1_instance_v1.npy',
                'A025-1_instance_v1.npy',
                'A041-1_instance_v2.npy'
            ]
            # use the above information to determine cur_brain_region id
            for cur_brain_region in range(len(brain_regions_names)):
                del cocoEval
                for filename, img_id in full_filename_to_id.items():
                    if filename not in target_filenames:
                        continue
                    print(filename, ' ', img_id)
                    print('Number of ground truths: ', len(coco.getAnnIds(imgIds=[img_id])))
                    # only process coco_dets in particular brain region
                    regions_final_result = brain_regions_filter_results(result_file, img_id, filename, cur_brain_region)
                    if len(regions_final_result) == 0:
                        if len(coco.getAnnIds(imgIds=[img_id])) == 0:
                            # 100% AP
                            print('Both ground truth and predicted bounding box are 0. Hence, Average Precision=100%')
                        else:
                            # All lesions are missdetected
                            print('0 Predicted bounding box but ground truth has at least one bounding box. Hence, Average Precision=0%')
                        continue
                    else:
                        if len(coco.getAnnIds(imgIds=[img_id])) == 0:
                            # N/A AP False positive
                            print('No ground truth in this brain region. Number of false positive: ', len(regions_final_result))
                        coco_dets_brain_region = coco.loadRes3D(regions_final_result)
                    cocoEval = COCOeval(coco, coco_dets_brain_region, iou_type, is3D=True)
                    # cocoEval = COCOeval(coco, coco_dets, iou_type, is3D=True) # original code
                    cocoEval.params.imgIds = [img_id]
                    cocoEval.evaluate(target_id=img_id)
                    cocoEval.accumulate()
                    cocoEval.summarize()
                    del cocoEval

            ############# output each patient's individual result #############
            # del cocoEval
            # for filename, img_id in full_filename_to_id.items():
            #     print(filename, ' ', img_id)
            #     cocoEval = COCOeval(coco, coco_dets, iou_type, is3D=True)
            #     cocoEval.params.imgIds = [img_id]
            #     cocoEval.evaluate(target_id=img_id)
            #     cocoEval.accumulate()
            #     cocoEval.summarize()
            #     del cocoEval

            ############## original implementation #############
            # cocoEval.params.imgIds = img_ids
            # if res_type == 'proposal':
            #     cocoEval.params.useCats = 0
            #     cocoEval.params.maxDets = list(max_dets)
            # cocoEval.evaluate(slice_label=slice_label)
            # cocoEval.accumulate()
            # cocoEval.summarize()

            # cocoEval.output_each_gt_best_results('intermediate-files/gt_best_1x.pickle') 
            # cocoEval.output_each_gt_best_results('intermediate-files/gt_best_1dot5x.pickle')
            # cocoEval.output_each_gt_best_results('intermediate-files/gt_best_1dot25x.pickle')
            # cocoEval.output_each_gt_best_results('intermediate-files/gt_best_combined.pickle') 
            # cocoEval.output_each_gt_best_results('intermediate-files/gt_best_combined-069.pickle') 


def fast_eval_recall(results,
                     coco,
                     max_dets,
                     iou_thrs=np.arange(0.5, 0.96, 0.05)):
    if mmcv.is_str(results):
        assert results.endswith('.pkl')
        results = mmcv.load(results)
    elif not isinstance(results, list):
        raise TypeError(
            'results must be a list of numpy arrays or a filename, not {}'.
            format(type(results)))

    gt_bboxes = []
    img_ids = coco.getImgIds()
    for i in range(len(img_ids)):
        ann_ids = coco.getAnnIds(imgIds=img_ids[i])
        ann_info = coco.loadAnns(ann_ids)
        if len(ann_info) == 0:
            gt_bboxes.append(np.zeros((0, 4)))
            continue
        bboxes = []
        for ann in ann_info:
            if ann.get('ignore', False) or ann['iscrowd']:
                continue
            x1, y1, w, h = ann['bbox']
            bboxes.append([x1, y1, x1 + w - 1, y1 + h - 1])
        bboxes = np.array(bboxes, dtype=np.float32)
        if bboxes.shape[0] == 0:
            bboxes = np.zeros((0, 4))
        gt_bboxes.append(bboxes)

    recalls = eval_recalls(
        gt_bboxes, results, max_dets, iou_thrs, print_summary=False)
    ar = recalls.mean(axis=1)
    return ar


def xyxy2xywh(bbox):
    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0] + 1,
        _bbox[3] - _bbox[1] + 1,
    ]


def proposal2json(dataset, results):
    json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        bboxes = results[idx]
        for i in range(bboxes.shape[0]):
            data = dict()
            data['image_id'] = img_id
            data['bbox'] = xyxy2xywh(bboxes[i])
            data['score'] = float(bboxes[i][4])
            data['category_id'] = 1
            json_results.append(data)
    return json_results


def det2json(dataset, results):
    json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        result = results[idx]
        for label in range(len(result)):
            bboxes = result[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = dataset.cat_ids[label]
                json_results.append(data)
    return json_results


def segm2json(dataset, results):
    json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        det, seg = results[idx]
        for label in range(len(det)):
            bboxes = det[label]
            segms = seg[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = dataset.cat_ids[label]
                segms[i]['counts'] = segms[i]['counts'].decode()
                data['segmentation'] = segms[i]
                json_results.append(data)
    return json_results


def results2json(dataset, results, out_file):
    if isinstance(results[0], list):
        json_results = det2json(dataset, results)
    elif isinstance(results[0], tuple):
        json_results = segm2json(dataset, results)
    elif isinstance(results[0], np.ndarray):
        json_results = proposal2json(dataset, results)
    else:
        raise TypeError('invalid type of results')
    mmcv.dump(json_results, out_file)


def segm2jsonRGB(dataset, results):
    json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        det_list, seg_list = results[idx]
        slice_num = 1
        for det, seg in zip(det_list, seg_list):
            for label in range(len(det)):
                bboxes = det[label]
                segms = seg[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = dataset.cat_ids[label]
                    segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]

                    if slice_num == 1:
                        data['slice_label'] = 'r'
                    elif slice_num == 2:
                        data['slice_label'] = 'g'
                    elif slice_num == 3:
                        data['slice_label'] = 'b'

                    json_results.append(data)
            slice_num += 1
    return json_results


def results2jsonRGB(dataset, results, out_file):
    if isinstance(results[0], list):
        json_results = det2json(dataset, results)
    elif isinstance(results[0], tuple):
        json_results = segm2jsonRGB(dataset, results)
    elif isinstance(results[0], np.ndarray):
        json_results = proposal2json(dataset, results)
    else:
        raise TypeError('invalid type of results')
    mmcv.dump(json_results, out_file)


def xyxyzz2xywhzd(bbox):
    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0] + 1,
        _bbox[3] - _bbox[1] + 1,
        _bbox[4],
        _bbox[5] - _bbox[4] + 1
    ]

# from https://zhuanlan.zhihu.com/p/54709759
def nms_3d_python(json_results, boxes, iou_thr):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    json_results = np.array(json_results)

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    z1 = boxes[:,4]
    z2 = boxes[:,5]
    cls_probs =boxes[:,6] 
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) * (z2 - z1 + 1)
    idxs = np.argsort(-cls_probs)
    keep = []
    while idxs.shape[0] > 0:
        i = idxs[0]
        keep.append(i)

        xx1 = np.clip(x1[idxs[1:]], a_min=x1[i], a_max=None)
        yy1 = np.clip(y1[idxs[1:]], a_min=y1[i], a_max=None)
        xx2 = np.clip(x2[idxs[1:]], a_min=None, a_max=x2[i])
        yy2 = np.clip(y2[idxs[1:]], a_min=None, a_max=y2[i])
        zz1 = np.clip(z1[idxs[1:]], a_min=z1[i], a_max=None)
        zz2 = np.clip(z2[idxs[1:]], a_min=None, a_max=z2[i])

        inter = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1) * np.maximum(0, zz2 - zz1 + 1)

        iou = inter / (areas[i]+areas[idxs[1:]]-inter) 
        idx = (iou <= iou_thr).nonzero()[0]
        if idx.shape[0] == 0:
            break
        idxs = idxs[idx+1]

    json_results = json_results[keep]   
    return json_results

def overlap_in_precomputed_proposals_inference_mode(bbox, filename, precomputed_proposals):
    proposals = np.array(precomputed_proposals[filename])
    overlaps = bbox_overlaps_3d(proposals, np.array([bbox]))
    if overlaps[:,0].max() != 0 :
        # there is at least one precomputed proposal that overlaps with filtered_result
        return True
    elif bbox[6] > 0.9:
        # if the confidence score is > 0.5, then it will still be considered.
        return True
    return False

def overlap_in_precomputed_proposals(full_id_to_filename, filtered_result, precomputed_proposals):
    proposals = np.array(precomputed_proposals[full_id_to_filename[filtered_result['image_id']].split('.')[0]])
    overlaps = bbox_overlaps_3d(proposals, np.array([filtered_result['original_bbox']]))
    if overlaps[:,0].max() != 0 :
        # there is at least one precomputed proposal that overlaps with filtered_result
        return True
    elif filtered_result['score'] > 0.9:
        # if the confidence score is > 0.5, then it will still be considered.
        return True
    return False

def apply_nms(full_filename_to_id, json_results, nms_thresh=0.1, score_thresh=0, filter_based_on_precomputed_proposals=False):
    if filter_based_on_precomputed_proposals:
        precomputed_proposals = np.load('precomputed-proposals1.5.pickle', allow_pickle=True)
        full_id_to_filename = dict()
        for filename, cur_id in full_filename_to_id.items():
            full_id_to_filename[cur_id] = filename

    # apply non-maximum suppresion to remove redundant and overlapping predictions
    nms_json_results = []
    for filename, img_id in full_filename_to_id.items():
        # get current patient's volume's predicted bounding boxes and their info
        cur_id_json_result = []
        cur_id_bboxes = []
        for json_result in json_results:
            if img_id == json_result['image_id']:
                cur_id_json_result.append(json_result)
                cur_id_bboxes.append(json_result['original_bbox'])
        cur_id_json_result = nms_3d_python(cur_id_json_result, cur_id_bboxes, nms_thresh)
        # add filtered results from NMS to final result, nms_json_results
        for filtered_result in cur_id_json_result:
            if filtered_result['score'] < score_thresh:
                continue
            if filter_based_on_precomputed_proposals and not overlap_in_precomputed_proposals(full_id_to_filename, filtered_result, precomputed_proposals):
                continue
            nms_json_results.append(filtered_result)

    return nms_json_results

def det2json3D(dataset, results, full_filename_to_id):
    json_results = []
    for idx in range(len(dataset)):
        result = results[idx]
        img_info = dataset.img_infos[idx]
        # check if full volume is provided, if it is, we use it for evaluation.
        if 'pos_top' in img_info:
            pos_top = img_info['pos_top']
            pos_left = img_info['pos_left']
            pos_front = img_info['pos_front']
            img_id = full_filename_to_id[img_info['orig_file_name']]
        else:
            img_id = dataset.img_ids[idx]

        for label in range(len(result)):
            bboxes = result[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                
                if 'pos_top' in img_info:
                    # convert local patch coordinates to global coordinates
                    bboxes[i][0] += pos_left
                    bboxes[i][1] += pos_top
                    bboxes[i][2] += pos_left
                    bboxes[i][3] += pos_top
                    bboxes[i][4] += pos_front
                    bboxes[i][5] += pos_front
                    
                data['bbox'] = xyxyzz2xywhzd(bboxes[i])
                data['score'] = float(bboxes[i][6])
                data['category_id'] = dataset.cat_ids[label]
                data['original_bbox'] = bboxes[i]
                json_results.append(data)

    json_results = apply_nms(full_filename_to_id, json_results, nms_thresh=0.1, score_thresh=0, filter_based_on_precomputed_proposals=False)
    return json_results

def det2json3DParcel(dataset, results, full_filename_to_id):
    json_results = []
    for idx in range(len(dataset)):
        result = results[idx]
        img_info = dataset.img_infos[idx]
        # check if full volume is provided, if it is, we use it for evaluation.
        if 'pos_top' in img_info:
            pos_top = img_info['pos_top']
            pos_left = img_info['pos_left']
            pos_front = img_info['pos_front']
            img_id = full_filename_to_id[img_info['orig_file_name']]
        else:
            img_id = dataset.img_ids[idx]
        
        result = list(result)
        parcellations = result.pop()
        for label in range(len(result)):
            bboxes = result[0][label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                
                if 'pos_top' in img_info:
                    # convert local patch coordinates to global coordinates
                    bboxes[i][0] += pos_left
                    bboxes[i][1] += pos_top
                    bboxes[i][2] += pos_left
                    bboxes[i][3] += pos_top
                    bboxes[i][4] += pos_front
                    bboxes[i][5] += pos_front
                    
                data['bbox'] = xyxyzz2xywhzd(bboxes[i])
                data['score'] = float(bboxes[i][6])
                data['category_id'] = dataset.cat_ids[label]
                data['original_bbox'] = bboxes[i]
                data['parcellations'] = parcellations[0][i].tolist()
                json_results.append(data)

    json_results = apply_nms(full_filename_to_id, json_results, nms_thresh=0.1, score_thresh=0, filter_based_on_precomputed_proposals=False)
    return json_results

'''
Perform translation from local patches to global volume of image
'''
def segm2json3D(dataset, results, full_filename_to_id):
    json_results = []
    seg_from_file = False
    for idx in range(len(dataset)):
    # for idx in range(6):
        det, seg = results[idx]
        if isinstance(seg, str):
            # original implementation: loading/converting 128x128x160 patches 
            seg = np.load(seg)['data']
            # new implementation: loading/converting potential CMB region
            # seg = np.load(seg, allow_pickle=True)
            seg_from_file = True

        img_info = dataset.img_infos[idx]
        # check if full volume is provided, if it is, we use it for evaluation.
        if 'pos_top' in img_info:
            pos_top = img_info['pos_top']
            pos_bottom = pos_top + seg[0][0].shape[1]
            pos_left = img_info['pos_left']
            pos_right = pos_left + seg[0][0].shape[2]
            pos_front = img_info['pos_front']
            pos_back = pos_front + seg[0][0].shape[0]
            img_id = full_filename_to_id[img_info['orig_file_name']]
        else:
            img_id = dataset.img_ids[idx]
        
        for label in range(len(det)):
            bboxes = det[label]
            segms = seg[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                
                if 'pos_top' in img_info:
                    # convert local patch coordinates to global coordinates
                    bboxes[i][0] += pos_left
                    bboxes[i][1] += pos_top
                    bboxes[i][2] += pos_left
                    bboxes[i][3] += pos_top
                    bboxes[i][4] += pos_front
                    bboxes[i][5] += pos_front

                    # original implementation: loading/converting 128x128x160 patches 
                    data['segm_pos_top'] = pos_top
                    data['segm_pos_bottom'] = pos_bottom
                    data['segm_pos_front'] = pos_front
                    data['segm_pos_back'] = pos_back
                    data['segm_pos_left'] = pos_left
                    data['segm_pos_right'] = pos_right

                    # new implementation: loading/converting potential CMB region, which is
                    # defined by bboxes[i]
                    # data['segm_pos_left'] = int(round(bboxes[i][0]))
                    # data['segm_pos_right'] = int(round(bboxes[i][2]))
                    # data['segm_pos_top'] = int(round(bboxes[i][1]))
                    # data['segm_pos_bottom'] = int(round(bboxes[i][3]))
                    # data['segm_pos_front'] = int(round(bboxes[i][4]))
                    # data['segm_pos_back'] = int(round(bboxes[i][5]))

                data['bbox'] = xyxyzz2xywhzd(bboxes[i])
                data['score'] = float(bboxes[i][6])
                data['category_id'] = dataset.cat_ids[label]
                if seg_from_file:
                    segm_out_filepath = 'in_progress/segmentation_{}_{}_{}.npz'.format(idx, label, i)
                    np.savez_compressed(segm_out_filepath, data=segms[i])
                    data['segmentation'] = segm_out_filepath
                else:
                    data['segmentation'] = segms[i]
                data['original_bbox'] = bboxes[i]
                json_results.append(data)

    json_results = apply_nms(full_filename_to_id, json_results)
    return json_results


def det2json3DMulti(dataset, dataset2, results, full_filename_to_id):
    json_results = []
    for idx in range(len(dataset)):
        result = results[idx]
        img_info = dataset.img_infos[idx]
        pos_top = img_info['pos_top']
        pos_left = img_info['pos_left']
        pos_front = img_info['pos_front']
        full_volume_img_id = full_filename_to_id[img_info['orig_file_name']]
        for label in range(len(result)):
            bboxes = result[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = full_volume_img_id
                
                # convert local patch coordinates to global coordinates
                bboxes[i][0] += pos_left
                bboxes[i][1] += pos_top
                bboxes[i][2] += pos_left
                bboxes[i][3] += pos_top
                bboxes[i][4] += pos_front
                bboxes[i][5] += pos_front
                
                data['bbox'] = xyxyzz2xywhzd(bboxes[i])
                data['score'] = float(bboxes[i][6])
                data['category_id'] = dataset.cat_ids[label]
                data['original_bbox'] = bboxes[i]
                json_results.append(data)
    
    for idx in range(len(dataset2)):
        result = results[idx]
        img_info = dataset2.img_infos[idx]
        pos_top = img_info['pos_top']
        pos_left = img_info['pos_left']
        pos_front = img_info['pos_front']
        full_volume_img_id = full_filename_to_id[img_info['orig_file_name']]
        for label in range(len(result)):
            bboxes = result[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = full_volume_img_id
                
                # convert local patch coordinates to global coordinates
                bboxes[i][0] += pos_left
                bboxes[i][1] += pos_top
                bboxes[i][2] += pos_left
                bboxes[i][3] += pos_top
                bboxes[i][4] += pos_front
                bboxes[i][5] += pos_front
                
                data['bbox'] = xyxyzz2xywhzd(bboxes[i])
                data['score'] = float(bboxes[i][6])
                data['category_id'] = dataset2.cat_ids[label]
                data['original_bbox'] = bboxes[i]
                json_results.append(data)

    json_results = apply_nms(full_filename_to_id, json_results)
    return json_results


def results2json3D(dataset, results, out_file, full_filename_to_id):
    if isinstance(results[0], list):
        json_results = det2json3D(dataset, results, full_filename_to_id)
    elif isinstance(results[0], tuple):
        json_results = segm2json3D(dataset, results, full_filename_to_id)
    else:
        raise TypeError('invalid type of results')
    dump_file = []
    for result in json_results:
        tmp = result.copy()
        tmp['original_bbox'] = 0
        if 'segmentation' in tmp:
            tmp['segmentation'] = 0
        dump_file.append(tmp)
    mmcv.dump(dump_file, out_file)
    return json_results


def results2json3DMulti(dataset, dataset2, results, out_file, full_filename_to_id):
    if isinstance(results[0], list):
        json_results = det2json3DMulti(dataset, dataset2, results, full_filename_to_id)
    elif isinstance(results[0], tuple):
        breakpoint()
        json_results = segm2json3D(dataset, results, full_filename_to_id)
    else:
        raise TypeError('invalid type of results')
    dump_file = []
    for result in json_results:
        tmp = result.copy()
        tmp['original_bbox'] = 0
        if 'segmentation' in tmp:
            tmp['segmentation'] = 0
        dump_file.append(tmp)
    mmcv.dump(dump_file, out_file)
    return json_results

def results2json3DParcel(dataset, results, out_file, full_filename_to_id):
    breakpoint()
    if len(results[0]) == 2:
        json_results = det2json3DParcel(dataset, results, full_filename_to_id)
    elif isinstance(results[0], tuple):
        json_results = segm2json3D(dataset, results, full_filename_to_id)
    else:
        raise TypeError('invalid type of results')

    dump_file = []
    for result in json_results:
        tmp = result.copy()
        tmp['original_bbox'] = 0
        if 'segmentation' in tmp:
            tmp['segmentation'] = 0
        dump_file.append(tmp)
    mmcv.dump(dump_file, out_file)
    return json_results