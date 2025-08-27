"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-12-03
@desc:   
"""

from typing import List, Tuple

import torch
from torch import Tensor
from panorama_coco.datasets.transforms import RandomRotateERP
from panorama_coco.registry import MODELS
from mmcv.ops import batched_nms
from mmdet.models import RetinaNet
from mmdet.structures import SampleList
from mmdet.structures.bbox import get_box_tensor
from mmengine.structures import InstanceData
from panorama_coco.structures.bbox.sphere_bboxes import SphereBoxes


@MODELS.register_module()
class SphRetinaNet(RetinaNet):
    """ Implementation of SphRetinaNet """ 
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
        if self.rotates:
            instances_list = []
            for _, rotate in enumerate(self.rotates):
                self._rotator.angles = rotate
                new_batch_inputs = []
                for image in batch_inputs:
                    new_image = self._rotator({'img': image})['img']
                    new_batch_inputs.append(new_image)
                new_batch_inputs = torch.stack(new_batch_inputs)
                x = self.extract_feat(new_batch_inputs)
                
                results_list = self.bbox_head.predict(
                    x, batch_data_samples
                )
                instances_list.append(results_list)
                
                # Restore predicted rotated boxes to their original orientation.
                for item in results_list:
                    item.bboxes.rotate(-rotate[0], -rotate[1], -rotate[2])
            
            # Apply NMS to all results.
            final_result_list = []
            cfg = self.bbox_head.test_cfg
            for ins_idx in range(len(instances_list[0])):
                bboxes_list = []
                scores_list = []
                lables_list = []
                for result_list in instances_list:
                    bboxes = get_box_tensor(result_list[ins_idx].bboxes) 
                    bboxes_list.append(bboxes)    
                    scores_list.append(result_list[ins_idx].scores)        
                    lables_list.append(result_list[ins_idx].labels)
                ins_bboxes = torch.cat(bboxes_list)
                ins_scores = torch.cat(scores_list)
                ins_labels = torch.cat(lables_list)
                
                det_bboxes, keep_idxs = batched_nms(
                    ins_bboxes, ins_scores,
                    ins_labels, cfg.nms
                )
                
                results = InstanceData()
                results.bboxes = SphereBoxes(
                    data=det_bboxes[..., :-1],
                    dtype=det_bboxes.dtype,
                    device=det_bboxes.device,
                )
                results.scores = det_bboxes[..., -1]
                results.labels = ins_labels[keep_idxs] 
                results = results[:cfg.max_per_img]
                final_result_list.append(results)
                 
            batch_data_samples = self.add_pred_to_datasample(
                batch_data_samples, final_result_list
            )
        else:
            batch_data_samples = super().predict(
                batch_inputs=batch_inputs,
                batch_data_samples=batch_data_samples,
                rescale=rescale
            )
        return batch_data_samples
    