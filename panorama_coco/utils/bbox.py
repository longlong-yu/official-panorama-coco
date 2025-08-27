"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-03-25
@desc:   
"""

from typing import List

import torch
from torch import Tensor

from panorama_coco.utils.spherical import ImageUtils, SphereImageUtils


def bbox_area(bbox: List[float]) -> float:
    """ Calculate area for bbox(x, y, w,h ) """
    return bbox[2] * bbox[3]


def r2hbox_xywh(rboxes: Tensor):
    """ rbox to hbox in xywh fashion. """
    w = rboxes[:, 2]
    h = rboxes[:, 3]
    a = rboxes[:, 4]
    cosa = torch.cos(a).abs()
    sina = torch.sin(a).abs()
    hboxes_w = cosa * w + sina * h
    hboxes_h = sina * w + cosa * h
    cx = rboxes[..., 0]
    cy = rboxes[..., 1]
    return torch.stack((cx, cy, hboxes_w, hboxes_h), -1)


def r2hbox_xyxy(rboxes: Tensor):
    """ rbox to hbox in xyxy fashion. """
    return ImageUtils.xywh2xyxy(r2hbox_xywh(rboxes))


def bfov2rbfov(bfovs):
    return torch.cat([bfovs, torch.zeros_like(bfovs[..., [0]])], dim=1)


def _sph2pix_box_transform(boxes, img_size):
    img_h, img_w = img_size
    boxes[..., 0] = (boxes[..., 0] / 360) * img_w
    boxes[..., 1] = (boxes[..., 1] / 180) * img_h
    boxes[..., 2] = (boxes[..., 2] / 360) * img_w
    boxes[..., 3] = (boxes[..., 3] / 180) * img_h
    return boxes


def _pix2sph_box_transform(boxes, img_size):
    img_h, img_w = img_size
    boxes[..., 0] = (boxes[..., 0] / img_w) * 360
    boxes[..., 1] =(boxes[..., 1] / img_h) * 180
    boxes[..., 2] = (boxes[..., 2] / img_w) * 360
    boxes[..., 3] = (boxes[..., 3] / img_h) * 180
    return boxes


class Sph2PlanarBoxTransform:
    def __init__(self, mode='sph2pix', box_version=4):
        assert mode in ['sph2pix']
        assert box_version in [4, 5]

        self.box_version = box_version
        self.transform = _sph2pix_box_transform
    
    def __call__(self, boxes, img_size=(512, 1024), box_version=None):
        box_version = self.box_version if box_version is None else box_version
        if box_version == 4:
            return ImageUtils.xywh2xyxy(self.transform(boxes, img_size))
        else:
            boxes = self.transform(boxes, img_size) #xywh
            boxes[..., 4] = -torch.deg2rad(boxes[..., 4]) #xywha
            return boxes


class Planar2SphBoxTransform:
    def __init__(self, mode='sph2pix', box_version=4):
        assert mode in ['sph2pix']
        assert box_version in [4, 5]

        self.box_version = box_version
        self.transform = _pix2sph_box_transform
    
    def __call__(self, boxes, img_size=(512, 1024), box_version=None):
        box_version = self.box_version if box_version is None else box_version
        if box_version == 4:
            rets =  self.transform(ImageUtils.xyxy2xywh(boxes), img_size) #xywh(bfov)
        else:
            _boxes = self.transform(ImageUtils.xyxy2xywh(boxes), img_size)
            rets = bfov2rbfov(_boxes) #xywha(rbfov)
        return SphereImageUtils.fov2sbox(rets)


def jiter_rotated_bboxes(bboxes1, bboxes2):
    eps = 1e-4 * 1.2345678
    Eps1 = torch.tensor([eps, eps, 2*eps, 2*eps, eps], device=bboxes1.device).unsqueeze_(0)
    Eps2 = torch.tensor([2*eps, 2*eps, eps, eps, 5*eps], device=bboxes1.device).unsqueeze_(0)
    similar_mask = (torch.abs(bboxes1[:, [0,2,3,4]] - bboxes2[:, [0,2,3,4]]) < eps).any(dim=1)
    bboxes1[similar_mask] += Eps1
    bboxes2[similar_mask] += Eps2

    eps = 1e-3 * 1.2345678
    angle_mask = torch.abs(bboxes1[:, 4] - bboxes2[:, 4]) < eps
    bboxes1[angle_mask, 4] += eps
    bboxes2[angle_mask, 4] += 2*eps

    pi = torch.pi
    bboxes1[:, 2:4].clamp_(min=2*eps/10)
    bboxes2[:, 2:4].clamp_(min=eps/10)
    bboxes1[:, 4].clamp_(min=-2*pi+2*eps, max=2*pi-eps)
    bboxes2[:, 4].clamp_(min=-2*pi+eps, max=2*pi-2*eps)

    return bboxes1, bboxes2

def jiter_spherical_bboxes(bboxes1, bboxes2):
    eps = 1e-4 * 1.2345678
    similar_mask = (torch.abs(bboxes1 - bboxes2) < eps).any(dim=1)
    bboxes1[similar_mask] = bboxes1[similar_mask] - 2* eps
    bboxes2[similar_mask] = bboxes2[similar_mask] + eps

    pi = 180
    torch.clamp_(bboxes1[:, 0], 2*eps, 2*pi-eps)
    torch.clamp_(bboxes1[:, 1:4], 2*eps, pi-eps)
    torch.clamp_(bboxes2[:, 0], eps, 2*pi-2*eps)
    torch.clamp_(bboxes2[:, 1:4], eps, pi-2*eps)
    if bboxes1.size(1) == 5:
        torch.clamp_(bboxes2[:, 4], -2*pi+eps, max=2*pi-2*eps)
        torch.clamp_(bboxes2[:, 4], -2*pi+2*eps, max=2*pi-eps)

    return bboxes1, bboxes2
