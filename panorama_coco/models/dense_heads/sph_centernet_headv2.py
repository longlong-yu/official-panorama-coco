"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-02-04
@desc:   
"""
from mmcv.cnn import Scale
from mmdet.models.utils import multi_apply
from mmdet.utils import ConfigType, InstanceList
from mmengine.model import normal_init
import torch
from torch import Tensor
import torch.nn as nn
from typing import List, Optional, Tuple

from panorama_coco.models.dense_heads.sph_centernet_head import SphCenterNetHead
from panorama_coco.registry import MODELS,TASK_UTILS
from panorama_coco.visualization.const import SampleItemType
from panorama_coco.visualization.visualizer import Visualizer


@MODELS.register_module()
class SphCenterNetHeadV2(SphCenterNetHead):

    def __init__(
        self,
        in_channels: int,
        feat_channels: int,
        loss_angle: ConfigType = dict(type='mmdet.L1Loss', loss_weight=1.),     

        scaled_angle: bool = True,
        angle_coder: ConfigType = dict(type='mmrotate.PseudoAngleCoder'),
        
        # 0: without iou_loss, 1: only with iou_loss, 2: with wh_loss, angle_loss and iou_loss
        with_iou_loss: int = 0,
        loss_iou: ConfigType = dict(type='TangentIouLoss', loss_weight=1.),
        **kwargs,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            feat_channels=feat_channels,
            with_iou_loss=with_iou_loss,
            loss_iou=loss_iou,
            **kwargs
        )
        self.angle_coder = TASK_UTILS.build(angle_coder)
        self.angle_head = self._build_head(
            in_channels, feat_channels, self.angle_coder.encode_size
        )
        
        self.loss_angle = MODELS.build(loss_angle)

        self.scaled_angle = scaled_angle
        if self.scaled_angle:
            self.angle_scale = Scale(1.0)
  
    def init_weights(self) -> None:
        """ Initialize the weights. """
        super().init_weights()
        for m in self.angle_head.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        

    def forward_single(self, x: Tensor) -> Tuple[Tensor, ...]:
        """
        Forward feature.

        Args:
            x (Tensor): Feature.

        Returns: Predict result.
        """
        center_heatmap_pred, wh_pred, offset_pred = super().forward_single(x)
        angle_pred = self.angle_coder.decode(self.angle_head(x))
        if self.scaled_angle:
            angle_pred = self.angle_scale(angle_pred).float()
        
        return center_heatmap_pred, wh_pred, angle_pred, offset_pred
  
    def loss_by_feat(
        self,
        ct_heatmap_preds: List[Tensor], 
        wh_preds: List[Tensor], 
        angle_preds: List[Tensor], 
        offset_preds: List[Tensor],
        target_result: dict,
        avg_factor: float,
        **kwargs,
    ) -> dict:
        """
        Calculate the loss based on the features extracted by the detection
        head.
        """
        assert len(ct_heatmap_preds) == len(wh_preds) == len(angle_preds) == len(offset_preds) == 1
        
        losses = multi_apply(
            self.loss_by_single_level,
            ct_heatmap_preds,
            wh_preds,
            angle_preds,
            offset_preds,
            target_result=target_result,
            avg_factor=avg_factor,
        )
        
        if self.with_iou_loss == 0:
            ret = dict(
                center_heatmap_loss=losses[0],
                wh_loss=losses[1],
                angle_loss=losses[2],
                offset_loss=losses[3],
            )
        elif self.with_iou_loss == 1:
            ret = dict(
                center_heatmap_loss=losses[0],
                iou_loss=losses[1],
                offset_loss=losses[2],
            )
        else:
           ret = dict(
                center_heatmap_loss=losses[0],
                wh_loss=losses[1],
                angle_loss=losses[2],
                iou_loss=losses[3],
                offset_loss=losses[4],
            ) 
        
        self._draw_feat(
            ct_heatmap_preds=ct_heatmap_preds,
            wh_preds=wh_preds,
            angle_preds=angle_preds,
            offset_preds=offset_preds,
            target_result=target_result
        )
           
        return ret
    
    def loss_by_single_level(
        self,
        ct_heatmap_pred: Tensor, 
        wh_pred: Tensor, 
        angle_pred: Tensor, 
        offset_pred: Tensor,
        target_result: dict,
        avg_factor: float,
        **kwargs,
    ) -> Tuple:
        """
        Calculate the loss based on the features extracted by the detection
        head.
        """      
        ct_heatmap_target = target_result['center_heatmap_target']
        wh_target = target_result['wh_target']
        angle_target = target_result['angle_target']
        offset_target = target_result['offset_target']
        wh_offset_target_weight = target_result['wh_offset_target_weight']
        
        erp_weights = self._erp_ct_weights.expand_as(ct_heatmap_target)
        
        ct_heatmap_loss = self.loss_center_heatmap(
            ct_heatmap_pred,
            ct_heatmap_target,
            weight=erp_weights,
            avg_factor=avg_factor,
        )
          
        wh_mask = wh_offset_target_weight > 0
        angle_mask = wh_offset_target_weight[:, [0]] > 0
        
        offset_loss = self.loss_offset(
            offset_pred[wh_mask],
            offset_target[wh_mask],
            avg_factor=avg_factor * 2,
        )
        
        ret_losses = [
            ct_heatmap_loss
        ]
        if self.with_iou_loss == 0 or self.with_iou_loss == 2: 
            wh_loss = self.loss_wh(
                wh_pred[wh_mask],
                wh_target[wh_mask],
                avg_factor=avg_factor * 2,
            )
            
            angle_loss = self.loss_angle(
                angle_pred[angle_mask],
                angle_target[angle_mask],
                avg_factor=avg_factor,
            )
            ret_losses += [wh_loss, angle_loss]
            
        if self.with_iou_loss == 1 or self.with_iou_loss == 2:
            bboxes = torch.cat(
                (wh_pred[wh_mask].view(2, -1), angle_pred[angle_mask].view(1, -1))
            ).permute(1, 0).reshape(-1, 3)
            bboxes_target = torch.cat(
                (wh_target[wh_mask].view(2, -1), angle_target[angle_mask].view(1, -1))
            ).permute(1, 0).reshape(-1, 3)
            iou_loss = self.loss_iou(
                bboxes,
                bboxes_target,
                avg_factor=avg_factor,
            )
            ret_losses += [iou_loss]
        
        ret_losses += [offset_loss]
        return ret_losses
    
    def predict_by_feat(
        self,
        ct_heatmap_preds: List[Tensor], 
        wh_preds: List[Tensor], 
        angle_preds: List[Tensor], 
        offset_preds: List[Tensor],
        batch_img_metas: Optional[List[dict]] = None,
        *,
        instances_list: List[InstanceList] = None,
        final: bool = True,
        **kwargs
    ) -> InstanceList:
        """
        Transform a batch of output features extracted from the head into
        bbox results.

        Args:
            features (Tuple[list[Tensor]]): Features.
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image. 
        """        
        rets = list(map(
            self._predict_by_feat_single,
            ct_heatmap_preds[-1],
            wh_preds[-1],
            angle_preds[-1],
            offset_preds[-1],
            batch_img_metas,
            instances_list if instances_list else [[]] * ct_heatmap_preds[-1].shape[0],
            [final] * ct_heatmap_preds[-1].shape[0],
        ))
        
        visualizer = Visualizer.get_current_instance()
        if visualizer.check_mode():
            for level in range(len(wh_preds)):
                visualizer.add_channel_items(
                    name=f'pred_heatmap_level{level}', 
                    values=ct_heatmap_preds[level].detach(), 
                    item_type=SampleItemType.HAETMAP,
            )
          
        return rets
