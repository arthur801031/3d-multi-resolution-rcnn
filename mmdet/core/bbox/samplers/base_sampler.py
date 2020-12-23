from abc import ABCMeta, abstractmethod

import torch

from .sampling_result import SamplingResult


class BaseSampler(metaclass=ABCMeta):

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 **kwargs):
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_gt_as_proposals = add_gt_as_proposals
        self.pos_sampler = self
        self.neg_sampler = self

    @abstractmethod
    def _sample_pos(self, assign_result, num_expected, **kwargs):
        pass

    @abstractmethod
    def _sample_neg(self, assign_result, num_expected, **kwargs):
        pass

    def sample(self,
               assign_result,
               bboxes,
               gt_bboxes,
               gt_labels=None,
               **kwargs):
        """Sample positive and negative bboxes.

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            bboxes (Tensor): Boxes to be sampled from.
            gt_bboxes (Tensor): Ground truth bboxes.
            gt_labels (Tensor, optional): Class labels of ground truth bboxes.

        Returns:
            :obj:`SamplingResult`: Sampling result.
        """
        if isinstance(gt_bboxes, list) and len(gt_bboxes) == 1:
            gt_bboxes = gt_bboxes[0]
        if isinstance(gt_labels, list) and len(gt_labels) == 1:
            gt_labels = gt_labels[0]

        # scores = None
        if bboxes.shape[1] >= 6:
            # if bboxes.shape[1] == 7:
            #     # sampling proposals for bbox head
            #     scores = bboxes[:, 6]
            bboxes = bboxes[:, :6]
        elif bboxes.shape[1] >= 4:
            bboxes = bboxes[:, :4]

        gt_flags = bboxes.new_zeros((bboxes.shape[0], ), dtype=torch.uint8)
        if self.add_gt_as_proposals:
            bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])

        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(
            assign_result, num_expected_pos, bboxes=bboxes, **kwargs)
        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.neg_sampler._sample_neg(
            assign_result, num_expected_neg, bboxes=bboxes, **kwargs)

        # Hard negative mining: half of neg_inds will be chosen based on the highest scores
        # sampling proposals for bbox head
        # if scores is not None:
        #     num_expected_neg_first_half = int(round(num_expected_neg / 2))
        #     _, topk_inds = scores.topk(num_expected_neg_first_half)
        #     topk_inds = topk_inds + gt_bboxes.shape[0] # account for gt_bboxes in the front, so push back index by number of gt_bboxes
        #     neg_inds = torch.cat((topk_inds, neg_inds))
        #     neg_inds = torch.unique(neg_inds.cpu(), sorted=False).to(neg_inds.device)
        #     # topk_inds begin at the back so flipping is required
        #     neg_inds = torch.flip(neg_inds, [0])
        #     neg_inds = neg_inds[:num_expected_neg]
        # else:
        #     neg_inds = neg_inds.unique()
        neg_inds = neg_inds.unique()

        return SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                              assign_result, gt_flags)
