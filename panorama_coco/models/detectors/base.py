"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-02-04
@desc:   
"""
from typing import Union
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.structures import SampleList
from torch import Tensor
from panorama_coco.registry import MODELS


@MODELS.register_module()
class BaseDetector(SingleStageDetector):
    """ Implementation of base detector. """ 
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def loss(
        self, 
        batch_inputs: Tensor,
        batch_data_samples: SampleList
    ) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as 'gt_bboxes', 'gt_bboxes_labels', 'sph_rot',.

        Returns:
            dict: A dictionary of loss components.
        """ 
        feat = self.extract_feat(batch_inputs)
        return self.bbox_head.loss(
            feat, batch_data_samples
        )
