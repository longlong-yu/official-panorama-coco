"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-09-22
@desc:   
"""
from typing import Union
from mmdet.models.task_modules import AssignResult, RandomSampler
from numpy import ndarray
from torch import Tensor
import torch
from panorama_coco.registry import TASK_UTILS


@TASK_UTILS.register_module()
class CommonSampler(RandomSampler):
    """
    A Sampler could keep pos/neg ratio.
    """

    def __init__(
        self,
        keep_ratio: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.keep_ratio = keep_ratio

    def _sample_pos(
        self, 
        assign_result: AssignResult, 
        num_expected: int,
        **kwargs
    ) -> Union[Tensor, ndarray]:
        """ """
        if self.keep_ratio:
            pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
            neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
            
            pos_num = int(pos_inds.numel() / self.pos_fraction)
            neg_num = int(neg_inds.numel() / (1 - self.pos_fraction))
            num_expected = int(min(pos_num, neg_num, self.num) * self.pos_fraction)

        return super()._sample_pos(
            assign_result=assign_result, 
            num_expected=num_expected,
            **kwargs
        )

    def _sample_neg(
        self, 
        assign_result: AssignResult, 
        num_expected: int,
        **kwargs
    ) -> Union[Tensor, ndarray]:
        """ """
        if self.keep_ratio:
            pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
            neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
            
            pos_num = int(pos_inds.numel() / self.pos_fraction)
            neg_num = int(neg_inds.numel() / (1 - self.pos_fraction))
            total = min(pos_num, neg_num, self.num)
            num_expected = total - int(total * self.pos_fraction)

        return super()._sample_neg(
            assign_result=assign_result, 
            num_expected=num_expected,
            **kwargs
        )
