import torch
import numpy as np

def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4)
        bboxes2 (Tensor): shape (n, 4), if is_aligned is ``True``, then m and n
            must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    """

    assert mode in ['iou', 'iof']

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        breakpoint()
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        if bboxes1.shape[1] == 6:
            xA = torch.max(bboxes1[:, None, 0], bboxes2[:, 0])
            yA = torch.max(bboxes1[:, None, 1], bboxes2[:, 1])
            xB = torch.min(bboxes1[:, None, 2], bboxes2[:, 2])
            yB = torch.min(bboxes1[:, None, 3], bboxes2[:, 3])
            zA = torch.max(bboxes1[:, None, 4], bboxes2[:, 4])
            zB = torch.min(bboxes1[:, None, 5], bboxes2[:, 5])

            interArea = (xB - xA + 1).clamp(min=0) * (yB - yA + 1).clamp(min=0) * (zB - zA + 1).clamp(min=0) 
            boxAArea = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1) * (bboxes1[:, 5] - bboxes1[:, 4] + 1)
            boxBArea = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1) * (bboxes2[:, 5] - bboxes2[:, 4] + 1)
            ious = interArea.to(dtype=torch.float) / (boxAArea[:, None] + boxBArea - interArea).to(dtype=torch.float)
        elif bboxes1.shape[1] == 4:
            lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
            rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

            wh = (rb - lt + 1).clamp(min=0)  # [rows, cols, 2]
            overlap = wh[:, :, 0] * wh[:, :, 1]
            area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
                bboxes1[:, 3] - bboxes1[:, 1] + 1)

            if mode == 'iou':
                area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                    bboxes2[:, 3] - bboxes2[:, 1] + 1)
                ious = overlap / (area1[:, None] + area2 - overlap)
            else:
                ious = overlap / (area1[:, None])
    return ious

'''
Test coordinates obtained from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
'''
def bbox_overlaps_test():
    bboxes1 = torch.tensor([[2, 3, 4, 6, 3, 4]])
    bboxes2 = torch.tensor([[2, 3, 4, 6, 3, 4]])
    assert round(bbox_overlaps(bboxes1, bboxes2).data.tolist()[0][0], 4) == 1.0

    bboxes1 = torch.tensor([[39, 63, 203, 112, 4, 5]])
    bboxes2 = torch.tensor([[54, 66, 198, 114, 4, 5]])
    assert round(bbox_overlaps(bboxes1, bboxes2).data.tolist()[0][0], 4) == 0.798

    bboxes1 = torch.tensor([[49, 75, 203, 125, 4, 5]])
    bboxes2 = torch.tensor([[42, 78, 186, 126, 4, 5]])
    assert round(bbox_overlaps(bboxes1, bboxes2).data.tolist()[0][0], 4) == 0.7899

    bboxes1 = torch.tensor([[31, 69, 201, 125, 4, 5]])
    bboxes2 = torch.tensor([[18, 63, 235, 135, 4, 5]])
    assert round(bbox_overlaps(bboxes1, bboxes2).data.tolist()[0][0], 4) == 0.6125

    bboxes1 = torch.tensor([[2, 3, 4, 6, 3, 4], [39, 63, 203, 112, 4, 5]])
    bboxes2 = torch.tensor([[2, 3, 4, 6, 3, 4], [54, 66, 198, 114, 4, 5], [49, 75, 203, 125, 4, 5]])
    result = bbox_overlaps(bboxes1, bboxes2)
    assert result.size(0) == 2 and result.size(1) == 3
    assert int(result[0][0]) == 1