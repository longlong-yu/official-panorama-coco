"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-10-16
@desc:   
"""
from mmdet.structures.bbox import register_box_converter
from torch import Tensor
import torch

from panorama_coco.utils.bbox import SphereImageUtils
from .sphere_bboxes import SphereBoxes, SphereHBoxes


@register_box_converter(SphereHBoxes, SphereBoxes)
def shbox2sbox(boxes: Tensor) -> Tensor:
    """Convert horizontal boxes to rotated boxes.

    Args:
        boxes (Tensor): horizontal box tensor with shape of (..., 4).

    Returns:
        Tensor: Rotated box tensor with shape of (..., 5).
    """
    return boxes


@register_box_converter(SphereBoxes, SphereHBoxes)
def sbox2shbox(boxes: Tensor) -> Tensor:
    """Convert horizontal boxes to rotated boxes.

    Args:
        boxes (Tensor): horizontal box tensor with shape of (..., 4).

    Returns:
        Tensor: Rotated box tensor with shape of (..., 5).
    """
    h_alpha, h_beta = SphereImageUtils.sbox_h_alpha_beta(boxes)
    return torch.cat([
        boxes[..., 0:2],
        h_alpha[..., None],
        h_beta[..., None],
        torch.zeros_like(h_alpha[..., None])
    ], dim=-1)
