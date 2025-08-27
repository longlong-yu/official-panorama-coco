# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from https://github.com/lilanxiao/Rotated_IoU/blob/master/box_intersection_2d.py  # noqa
# Adapted from https://github.com/lilanxiao/Rotated_IoU/blob/master/oriented_iou_loss.py  # noqa
from typing import Tuple

import torch
from torch import Tensor

from mmcv.ops.diff_iou_rotated import EPSILON, box2corners, box_in_box, build_vertices, sort_indices, calculate_area 


def box_intersection(corners1: Tensor,
                     corners2: Tensor) -> Tuple[Tensor, Tensor]:
    """Find intersection points of rectangles.
    Convention: if two edges are collinear, there is no intersection point.

    Args:
        corners1 (Tensor): (B, N, 4, 2) First batch of boxes.
        corners2 (Tensor): (B, N, 4, 2) Second batch of boxes.

    Returns:
        Tuple:
         - Tensor: (B, N, 4, 4, 2) Intersections.
         - Tensor: (B, N, 4, 4) Valid intersections mask.
    """
    # build edges from corners
    # B, N, 4, 4: Batch, Box, edge, point
    line1 = torch.cat([corners1, corners1[:, :, [1, 2, 3, 0], :]], dim=3)
    line2 = torch.cat([corners2, corners2[:, :, [1, 2, 3, 0], :]], dim=3)
    # duplicate data to pair each edges from the boxes
    # (B, N, 4, 4) -> (B, N, 4, 4, 4) : Batch, Box, edge1, edge2, point
    line1_ext = line1.unsqueeze(3)
    line2_ext = line2.unsqueeze(2)
    x1, y1, x2, y2 = line1_ext.split([1, 1, 1, 1], dim=-1)
    x3, y3, x4, y4 = line2_ext.split([1, 1, 1, 1], dim=-1)
    # math: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    numerator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    denumerator_t = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    t = denumerator_t / numerator
    t[numerator == .0] = -1.
    mask_t = (t > 0) & (t < 1)  # intersection on line segment 1
    denumerator_u = (x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)
    u = -denumerator_u / numerator
    u[numerator == .0] = -1.
    mask_u = (u > 0) & (u < 1)  # intersection on line segment 2
    mask = mask_t * mask_u
    
    # overwrite with EPSILON. otherwise numerically unstable
    # !Note: fix the case when numerator + EPSILON == 0
    # t = denumerator_t / (numerator + EPSILON)
    numerator = torch.where(numerator + EPSILON == 0, numerator, numerator + EPSILON)
    t = denumerator_t / numerator
    
    intersections = torch.stack([x1 + t * (x2 - x1), y1 + t * (y2 - y1)],
                                dim=-1)
    intersections = intersections * mask.float().unsqueeze(-1)
    return intersections, mask


def oriented_box_intersection_2d(corners1: Tensor,
                                 corners2: Tensor) -> Tuple[Tensor, Tensor]:
    """Calculate intersection area of 2d rotated boxes.

    Args:
        corners1 (Tensor): (B, N, 4, 2) First batch of boxes.
        corners2 (Tensor): (B, N, 4, 2) Second batch of boxes.

    Returns:
        Tuple:
         - Tensor (B, N): Area of intersection.
         - Tensor (B, N, 9, 2): Vertices of polygon with zero padding.
    """
    intersections, valid_mask = box_intersection(corners1, corners2)
    c12, c21 = box_in_box(corners1, corners2)
    vertices, mask = build_vertices(corners1, corners2, c12, c21,
                                    intersections, valid_mask)
    sorted_indices = sort_indices(vertices, mask)
    return calculate_area(sorted_indices, vertices)


def diff_iou_rotated_2d(box1: Tensor, box2: Tensor) -> Tensor:
    """Calculate differentiable iou of rotated 2d boxes.

    Args:
        box1 (Tensor): (B, N, 5) First box.
        box2 (Tensor): (B, N, 5) Second box.

    Returns:
        Tensor: (B, N) IoU.
    """
    corners1 = box2corners(box1)
    corners2 = box2corners(box2)
    intersection, _ = oriented_box_intersection_2d(corners1,
                                                   corners2)  # (B, N)
    area1 = box1[:, :, 2] * box1[:, :, 3]
    area2 = box2[:, :, 2] * box2[:, :, 3]
    union = area1 + area2 - intersection
    iou = intersection / union
    return iou
