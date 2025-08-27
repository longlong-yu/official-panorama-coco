"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-03-25
@desc:   
"""
from mmdet.datasets.api_wrappers import COCOeval as MM_COCOEval
from mmrotate.evaluation.metrics.rotated_coco_metric import RotatedCocoEval as MM_RotatedCocoEval
import numpy as np
import torch
from panorama_coco.core.iou import UnbiasedIoU
from panorama_coco.registry import TASK_UTILS


@TASK_UTILS.register_module()
class COCOEval(MM_COCOEval):
    """ Just a wrapper of COCOeval for TASK_UTILS. """
    pass

@TASK_UTILS.register_module()
class RotatedCocoEval(MM_RotatedCocoEval):
    """ Just a wrapper of RotatedCocoEval for TASK_UTILS. """
    pass


@TASK_UTILS.register_module()
class PandoraEval(COCOEval):
    """This is a wrapper to support unbiased panoramic iou Eval."""
    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]

        if p.iouType == 'bbox':
            # Modified for Rotated Box
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
            # Convert List[List[float]] to Tensor for iou compute
            with torch.no_grad():
                g = torch.tensor(g)
                d = torch.tensor(d)
                ious = UnbiasedIoU.cross_iou(d, g)
        else:
            raise Exception('unknown iouType for iou computation')

        return ious
