"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-08-04
@desc:   refer: Sph2Pob
"""
import torch
from mmdet.models.task_modules import AnchorGenerator
from mmdet.structures.bbox import get_box_type
from panorama_coco.registry import TASK_UTILS
from panorama_coco.utils.bbox import Planar2SphBoxTransform


@TASK_UTILS.register_module()
class SphAnchorGenerator(AnchorGenerator):
    """Spherical anchor generator for 2D anchor-based detectors.

    Horizontal bounding box represented by (theta, phi, alpha, beta).
    """
    def __init__(
        self, 
        box_formator='sph2pix', 
        box_version=4,
        box_type: str = 'sbox',
        *args,
        **kwargs
    ):
        super(SphAnchorGenerator, self).__init__(*args, **kwargs)
        assert box_formator in ['sph2pix', 'pix2sph', 'sph2tan', 'tan2sph']
        assert box_version in [4, 5]
        self.box_formator = Planar2SphBoxTransform(box_formator, box_version)
        self.box_type = box_type


    def single_level_grid_priors(
        self,
        featmap_size,
        level_idx,
        dtype=torch.float32,
        device='cuda'
    ):
        anchors = super().single_level_grid_priors(featmap_size, level_idx, dtype, device)
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        img_h, img_w = feat_h * stride_h, feat_w * stride_w

        anchors = self.box_formator(anchors, (img_h, img_w))
        _, cls_type = get_box_type(self.box_type)
        return cls_type(anchors)
        

    def single_level_grid_anchors(
        self,
        base_anchors,
        featmap_size,
        stride=(16, 16),
        device='cuda'
    ):
        raise Exception('single_level_grid_anchors is not supported any more!')
 