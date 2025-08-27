"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-03-25
@desc:   
"""
from typing import List, Tuple
from mmdet.structures import DetDataSample
from panorama_coco.evaluation.metrics.coco_metric import CocoMetric
from panorama_coco.registry import METRICS


@METRICS.register_module()
class PandoraMetric(CocoMetric):
    """
    Pandora evaluation metric.
    """
    def __init__(
        self,
        *,
        coco_eval_config: dict = dict(type='PandoraEval'), 
        **kwargs
    ) -> None:
        super().__init__(
            coco_eval_config=coco_eval_config,
            **kwargs
        )
    
    @staticmethod
    def prepare_sample(sample: DetDataSample):
        return sample    
    
    def prepare(self, gts: List[dict], preds: List[dict]) -> Tuple[List[dict], List[dict]]:
        for gt in gts:
            for ann in gt['anns']:
                # Skip actual area computation; set the value to 1 as a placeholder
                ann['area'] = 1
        for pred in preds:
            for ann in pred['anns']:
                ann['area'] = 1
                
        return gts, preds
