"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-08-29
@desc:   
"""
from typing import List, Optional, Tuple
from mmdet.models import BaseRoIExtractor
from mmdet.utils import ConfigType, OptMultiConfig
from torch import Tensor
import torch

from panorama_coco.registry import MODELS


@MODELS.register_module()
class SingleRoIExtractor(BaseRoIExtractor):
    """
    refer MMdet.models.SingleRoIExtractor.
    Adapte map_roi_levles() to ERP/FOV.
    """
    def __init__(
        self,
        roi_layer: ConfigType,
        out_channels: int,
        featmap_strides: List[int],
        finest_scale: float = 0.082, # 75° / 16
        init_cfg: OptMultiConfig = None
    ) -> None:
        super().__init__(
            roi_layer=roi_layer,
            out_channels=out_channels,
            featmap_strides=featmap_strides,
            init_cfg=init_cfg)
        self.finest_scale = finest_scale
    
    
    def map_roi_levels(
        self, 
        origin_rois: Tensor,
        num_levels: int,
    ) -> Tensor:
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt(origin_rois[:, 3] * origin_rois[:, 4])
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        return target_lvls.clamp(min=0, max=num_levels - 1).long()
    
    def forward(
        self,
        feats: Tuple[Tensor],
        rois: Tensor,
        origin_rois: Tensor,
        roi_scale_factor: Optional[float] = None
    ):
        """Extractor ROI feats.

        Args:
            feats (Tuple[Tensor]): Multi-scale features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.
            roi_scale_factor (Optional[float]): RoI scale factor.
                Defaults to None.

        Returns:
            Tensor: RoI feature.
        """
        # convert fp32 to fp16 when amp is on
        rois = rois.type_as(feats[0])
        out_size = self.roi_layers[0].output_size
        num_levels = len(feats)
        roi_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels, *out_size)

        # TODO: remove this when parrots supports
        if torch.__version__ == 'parrots':
            roi_feats.requires_grad = True

        if num_levels == 1:
            if len(rois) == 0:
                return roi_feats
            return self.roi_layers[0](feats[0], rois)

        target_lvls = self.map_roi_levels(origin_rois, num_levels)

        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        for i in range(num_levels):
            mask = target_lvls == i
            inds = mask.nonzero(as_tuple=False).squeeze(1)
            if inds.numel() > 0:
                rois_ = rois[inds]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] = roi_feats_t
            else:
                # Sometimes some pyramid levels will not be used for RoI
                # feature extraction and this will cause an incomplete
                # computation graph in one GPU, which is different from those
                # in other GPUs and will cause a hanging error.
                # Therefore, we add it to ensure each feature pyramid is
                # included in the computation graph to avoid runtime bugs.
                roi_feats += sum(
                    x.view(-1)[0]
                    for x in self.parameters()) * 0. + feats[i].sum() * 0.
        return roi_feats
