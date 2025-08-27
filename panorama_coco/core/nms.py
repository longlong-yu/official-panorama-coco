"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-06-24
@desc:   
"""
from typing import Tuple
from mmcv.ops.nms import array_like_type
from numba import jit
import numpy as np
import torch
from torch import Tensor
# !Note: In unbiased_iou, the angle-to-radian conversion should be removed, and 
# the coordinate range of the center point needs to be adjusted.
from panorama_coco.sdk.sph2pob.bbox.iou import unbiased_iou, naive_iou, sph2pob_efficient_iou # noqa


@jit(nopython=True)
def _obtain_keep(iou: np.ndarray, B: np.ndarray, iou_threshold: float):
    keep = []

    while B.size > 0:
        keep.append(B[0])
        if B.size == 1:
            break
        tmp_iou = iou[B[0], B[1:]]
        inds = np.nonzero(tmp_iou <= iou_threshold)[0].reshape(-1)
        B = B[inds + 1]
    return keep


@torch.jit.script
def _obtain_keep_gpu(iou: Tensor, B: Tensor, iou_threshold: float):
    keep = []

    while B.shape[0] > 0:
        keep.append(B[0])
        if B.shape[0] == 1:
            break
        tmp_iou = iou[B[0], B[1:]]
        inds = torch.nonzero(tmp_iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return keep


def sph_nms_op(boxes, scores, iou_threshold, iou_calculator, with_cpu: bool = True):
    var_dim = boxes.shape[1]
    assert len(boxes.shape) == 2 and var_dim in [4, 5]
    
    if with_cpu:
        B = torch.argsort(scores, descending=True).cpu().numpy()
        iou = iou_calculator(
            boxes,
            boxes,
        ).cpu().numpy()
        keep = _obtain_keep(
            iou=iou,
            B=B,
            iou_threshold=iou_threshold
        )
    else:
        B = torch.argsort(scores, descending=True)
        iou = iou_calculator(
            boxes,
            boxes,
        )
        keep = _obtain_keep_gpu(
            iou=iou,
            B=B,
            iou_threshold=iou_threshold
        )
    
    return torch.tensor(keep, device=boxes.device)


def sph2pob_nms(
    boxes: array_like_type,
    scores: array_like_type,
    *,
    iou_threshold: float = 0.5,
    iou_calculator: str = 'sph2pob_efficient_iou',
    with_cpu: bool = True,
    **kwargs
) -> Tuple[array_like_type, array_like_type]:
    if isinstance(iou_calculator, str):
        iou_calculator = eval(iou_calculator)
    # !Note: sph2pob uses spherical coordinates: [0, 360] longitude, [0, 180] latitude, origin at top-left
    boxes[..., 0] = boxes[..., 0] + torch.pi
    boxes[..., 1] = torch.pi / 2 - boxes[..., 1]
    
    with torch.no_grad():
        keep = sph_nms_op(boxes, scores, iou_threshold, iou_calculator, with_cpu)
        
    boxes[..., 0] = boxes[..., 0] - torch.pi
    boxes[..., 1] = torch.pi / 2 - boxes[..., 1]
    
    boxes = torch.cat((boxes[keep], scores[keep][..., None]), dim=-1)
    return boxes, keep
