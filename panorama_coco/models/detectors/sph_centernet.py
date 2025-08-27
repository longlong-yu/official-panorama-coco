"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-04-19
@desc:   
"""
from typing import List, Tuple, Union
from mmdet.structures import SampleList
from torch import Tensor
import torch
from panorama_coco.datasets.transforms.transforms import RandomRotateERP
from panorama_coco.models.detectors.base import BaseDetector
from panorama_coco.registry import MODELS
from panorama_coco.structures.bbox.sphere_bboxes import SphereBoxes


@MODELS.register_module()
class SphCenterNet(BaseDetector):
    """Implementation of H2SRBox""" 
    def __init__(
        self,
        rotates: List[Tuple[float, float, float]] = [],
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.rotates = []
        for item in rotates:
            self.rotates.append((item[0]* 2 * torch.pi, item[1] * 2 * torch.pi, item[2] * 2 * torch.pi))
        self._rotator = RandomRotateERP(need_permute=False)

    def predict(
        self,
        batch_inputs: Tensor,
        batch_data_samples: SampleList,
        rescale: bool = True
    ) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        if self.rotates:
            instances_list = []
            for i, rotate in enumerate(self.rotates):
                self._rotator.angles = rotate
                new_batch_inputs = []
                for image in batch_inputs:
                    new_image = self._rotator({'img': image})['img']
                    new_batch_inputs.append(new_image)
                new_batch_inputs = torch.stack(new_batch_inputs)
                x = self.extract_feat(new_batch_inputs)
                if i == len(self.rotates) - 1:
                    # Rotate the previous predicted boxes to align with the current orientation.
                    for results_list in instances_list:
                        for item in results_list:
                            bboxes = SphereBoxes(
                                data=item.bboxes,
                                dtype=item.bboxes.dtype,
                                device=item.bboxes.device,
                            )
                            bboxes.rotate(*rotate)
                            item.bboxes = bboxes.tensor
                    
                    results_list = self.bbox_head.predict(
                        x, batch_data_samples, 
                        instances_list=list(zip(*instances_list)), 
                        final=True
                    )
                else:
                    results_list = self.bbox_head.predict(
                        x, batch_data_samples, final=False
                    )
                    instances_list.append(results_list)
                
                # Restore the predicted rotated boxes to their original (unrotated) state.
                for item in results_list:
                    bboxes = SphereBoxes(
                        data=item.bboxes,
                        dtype=item.bboxes.dtype,
                        device=item.bboxes.device,
                    )
                    bboxes.rotate(-rotate[0], -rotate[1], -rotate[2])
                    item.bboxes = bboxes.tensor
            
            batch_data_samples = self.add_pred_to_datasample(
                batch_data_samples, results_list
            )
        else:
            batch_data_samples = super().predict(
                batch_inputs=batch_inputs,
                batch_data_samples=batch_data_samples,
                rescale=rescale
            )
        return batch_data_samples
    