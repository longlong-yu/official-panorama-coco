"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-02-04
@desc:   
"""
from mmcv.cnn import Scale
from mmcv.ops import batched_nms
from mmdet.models.utils import  get_local_maximum, multi_apply, transpose_and_gather_feat
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, InstanceList
from mmengine.model import bias_init_with_prob, normal_init
from mmengine.structures import InstanceData
from mmengine.visualization import Visualizer
import torch
from torch import Tensor
import torch.nn as nn
from typing import List, Optional, Tuple

from panorama_coco.core.device import torch_device
from panorama_coco.core.head import BaseHead
from panorama_coco.core.nms import sph2pob_nms # noqa
from panorama_coco.registry import MODELS,TASK_UTILS
from panorama_coco.utils.spherical import SphereImageUtils
from panorama_coco.visualization.const import SampleItemType


@MODELS.register_module()
class SphCenterNetHead(BaseHead):

    def __init__(self,
        erp_shape: Tuple[int, int],
        in_channels: int,
        feat_channels: int,
        num_classes: int,
        loss_center_heatmap: ConfigType = dict(
            type='mmdet.GaussianFocalLoss',
            loss_weight=1.
        ),
        loss_wh: ConfigType = dict(type='mmdet.L1Loss', loss_weight=10.),
        loss_offset: ConfigType = dict(type='mmdet.L1Loss', loss_weight=1.),        

        wh_angle_coder: ConfigType = dict(type='mmrotate.PseudoAngleCoder'),
        
        threshold: float = 0.3,
        auto_ajust: bool = False,
        with_erp_ct_weight: bool = False,
        
        # 0: without iou_loss, 1: only with iou_loss, 2: with wh_loss and iou_loss
        with_iou_loss: int = 0,
        # Can't choose TangentIouLoss
        loss_iou: ConfigType = dict(type='TangentIouLoss', loss_weight=1.),
        
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.wh_angle_coder = TASK_UTILS.build(wh_angle_coder)
        
        self.heatmap_head = self._build_head(
            in_channels, feat_channels, num_classes
        )
        self.wh_head = self._build_head(
            in_channels, feat_channels, self.wh_angle_coder.encode_size * 2
        )
        self.offset_head = self._build_head(in_channels, feat_channels, 2)
        
        self.loss_center_heatmap = MODELS.build(loss_center_heatmap)
        self.loss_wh = MODELS.build(loss_wh)
        self.loss_offset = MODELS.build(loss_offset)

        self.threshold = threshold
        self.auto_ajust = auto_ajust
        
        self.erp_shape = (erp_shape[0] // 4, erp_shape[1] // 4)
        self.with_erp_ct_weight = with_erp_ct_weight
        self._erp_ct_weights = self._build_erp_ct_weights(
            erp_h=self.erp_shape[0],
            erp_w=self.erp_shape[1],
        )
        
        self.with_iou_loss = with_iou_loss
        self.loss_iou = MODELS.build(loss_iou)
        
        if self.test_cfg.with_nms and self.test_cfg.need_eval:
            self.test_cfg.nms.type = eval(self.test_cfg.nms.type)
 
    @staticmethod
    def _build_head(
        in_channels: int, 
        feat_channels: int,
        out_channels: int
    ) -> nn.Sequential:
        """Build head for each branch."""
        return nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, out_channels, kernel_size=1)
        )

    def _build_erp_ct_weights(self, erp_h: int, erp_w: int) -> Tensor:
        device = torch_device()
        y = torch.linspace(0.5, -0.5, erp_h, device=device)
        x = torch.linspace(-1, 1, erp_w, device=device)
        y, _ = torch.meshgrid([y, x], indexing='ij')
        weights = (torch.cos(y * torch.pi / erp_h) - torch.cos((y + 1) * torch.pi / erp_h)) * 2 * torch.pi / erp_w
        if self.with_erp_ct_weight:
            return weights
        else:
            return torch.ones_like(weights)
    
    def init_weights(self) -> None:
        """ Initialize the weights. """
        bias_init = bias_init_with_prob(0.1)
        self.heatmap_head[-1].bias.data.fill_(bias_init)
        for head in [self.wh_head, self.offset_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)
        super().init_weights()

    def forward_single(self, x: Tensor) -> Tuple[Tensor, ...]:
        """
        Forward feature.

        Args:
            x (Tensor): Feature.

        Returns: Predict result.
        """
        center_heatmap_pred = self.heatmap_head(x).sigmoid()
        wh_pred = self.wh_angle_coder.decode(self.wh_head(x))
        offset_pred = self.offset_head(x)
        
        return center_heatmap_pred, wh_pred, offset_pred

    def get_targets(
        self, 
        batch_data_samples: SampleList,
        *, 
        feats: Tuple[Tensor],
        **kwargs
    ) -> Tuple[dict, int]:
        """
        Compute targets in multiple images.

        Args:
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            tuple[dict, float]: The float value is mean avg_factor, the dict
            has components below:
               - center_heatmap_target (Tensor): targets of center heatmap,
                   shape (B, num_classes, H, W).
               - wh_target (Tensor): targets of wh predict, shape
                   (B, 2, H, W).
               - offset_target (Tensor): targets of offset predict, shape
                   (B, 2, H, W).
               - wh_offset_target_weight (Tensor): weights of wh and offset
                   predict, shape (B, 2, H, W).
        """
        target_h, target_w = feats[0][0].shape[-2:]
        
        center_heatmap_targets = []
        wh_targets = []
        angle_targets = []
        offset_targets = []
        wh_offset_target_weights = []
        avg_factor = 0
        for sample in batch_data_samples:
            gt_instances = sample.gt_instances
            
            center_heatmap_target = gt_instances[0].bboxes.new_zeros([
                self.num_classes, target_h, target_w 
            ])
            wh_target = center_heatmap_target.new_zeros([
                2, target_h, target_w
            ])
            angle_target = center_heatmap_target.new_zeros([
                1, target_h, target_w
            ])
            offset_target = center_heatmap_target.new_zeros([
                2, target_h, target_w
            ])
            wh_offset_target_weight = center_heatmap_target.new_zeros([
                2, target_h, target_w
            ])
            
            gt_bboxes = gt_instances.bboxes.tensor
            gt_labeles = gt_instances.labels
            
            for gt_bbox, gt_label in zip(gt_bboxes, gt_labeles):
                SphereImageUtils.gen_gaussian_target(
                    center_heatmap=center_heatmap_target[gt_label],
                    sboxes=gt_bbox[None],
                    threshold=self.threshold,
                    auto_ajust=self.auto_ajust
                )
                
                erp_ctx, erp_cty = SphereImageUtils.theta_phi2pxpy(
                    theta_phi=gt_bbox[0:2],
                    w=target_w,
                    h=target_h,
                    clamp=False,
                )
                erp_ctx_int = erp_ctx.int()
                erp_cty_int = erp_cty.int()
                
                wh_target[0, erp_cty_int, erp_ctx_int] = gt_bbox[2]
                wh_target[1, erp_cty_int, erp_ctx_int] = gt_bbox[3]
                angle_target[0, erp_cty_int, erp_ctx_int] = gt_bbox[4]
                offset_target[0, erp_cty_int, erp_ctx_int] = erp_ctx - erp_ctx_int
                offset_target[1, erp_cty_int, erp_ctx_int] = erp_cty - erp_cty_int

                wh_offset_target_weight[:, erp_cty_int, erp_ctx_int] = 1
                
                avg_factor += 1
            
            center_heatmap_targets.append(center_heatmap_target)
            wh_targets.append(wh_target)
            angle_targets.append(angle_target)
            offset_targets.append(offset_target)
            wh_offset_target_weights.append(wh_offset_target_weight)
        
        avg_factor = max(1, avg_factor)  
        target_result = dict(
            center_heatmap_target=torch.stack(center_heatmap_targets),
            wh_target=torch.stack(wh_targets),
            angle_target=torch.stack(angle_targets),
            offset_target=torch.stack(offset_targets),
            wh_offset_target_weight=torch.stack(wh_offset_target_weights),
        )
        return target_result, avg_factor
    
    def loss_by_feat(
        self,
        ct_heatmap_preds: List[Tensor], 
        wh_preds: List[Tensor],
        offset_preds: List[Tensor],
        target_result: dict,
        avg_factor: float,
        **kwargs,
    ) -> dict:
        """
        Calculate the loss based on the features extracted by the detection
        head.
        """
        assert len(ct_heatmap_preds) == len(wh_preds) == len(offset_preds) == 1
        
        losses = multi_apply(
            self.loss_by_single_level,
            ct_heatmap_preds,
            wh_preds,
            offset_preds,
            target_result=target_result,
            avg_factor=avg_factor,
        )
        
        if self.with_iou_loss == 0:
            ret = dict(
                center_heatmap_loss=losses[0],
                wh_loss=losses[1],
                offset_loss=losses[2],
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
                iou_loss=losses[2],
                offset_loss=losses[3],
            ) 
        
        self._draw_feat(
            ct_heatmap_preds=ct_heatmap_preds,
            wh_preds=wh_preds,
            angle_preds=[torch.zeros_like(item[:, [0], ...]) for item in wh_preds],
            offset_preds=offset_preds,
            target_result=target_result
        )
           
        return ret
    
    def _draw_feat(
        self,
        ct_heatmap_preds: List[Tensor], 
        wh_preds: List[Tensor],
        angle_preds: List[Tensor],
        offset_preds: List[Tensor],
        target_result: dict,
    ):
        """ Intermediate feature visualization. """
        visualizer = Visualizer.get_current_instance()
        if not visualizer.check_mode():
            return

        wh_target = target_result['wh_target']
        ct_heatmap_target = target_result['center_heatmap_target'] 
        for level in range(len(wh_preds)):
            for batch_idx in range(wh_preds[level].shape[0]):
                wh_mask = wh_target[batch_idx, 0] > 0
                # (N, 2), (w, h)
                wh = wh_preds[level][batch_idx, :, wh_mask].permute(1, 0)
                angle = angle_preds[level][batch_idx, :, wh_mask].permute(1, 0)
                # (2, N), (theta, phi)
                erp_pxpy = wh_mask.nonzero().permute(1, 0)[[1, 0], ...]
                theta_phi = SphereImageUtils.pxpy2theta_phi(erp_pxpy, w=self.erp_shape[1], h=self.erp_shape[0])
                visualizer.add_channel_item(
                    name=f'pred_sbox_level{level}', 
                    value=torch.cat([theta_phi.permute(1, 0), wh, angle], dim=-1).detach(), 
                    item_type=SampleItemType.SBOX,
                )
                
            visualizer.add_channel_items(
                name=f'pred_heatmap_level{level}', 
                values=ct_heatmap_preds[level].detach(), 
                item_type=SampleItemType.HAETMAP,
            )
            if level == 0:
                visualizer.add_channel_items(
                    name=f'gt_heatmap', 
                    values=ct_heatmap_target.detach(), 
                    item_type=SampleItemType.HAETMAP,
                )

    def loss_by_single_level(
        self,
        ct_heatmap_pred: Tensor,
        wh_pred: Tensor,
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
            ret_losses += [wh_loss]
            
        if self.with_iou_loss == 1 or self.with_iou_loss == 2:
            bboxes = wh_pred[wh_mask].view(2, -1).permute(1, 0).reshape(-1, 2)
            bboxes_target = wh_target[wh_mask].view(2, -1).permute(1, 0).reshape(-1, 2)
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
        return list(map(
            self._predict_by_feat_single,
            ct_heatmap_preds[-1],
            wh_preds[-1],
            torch.zeros_like(wh_preds[-1][:, [0], ...]),
            offset_preds[-1],
            batch_img_metas,
            instances_list if instances_list else [[]] * ct_heatmap_preds[-1].shape[0],
            [final] * ct_heatmap_preds[-1].shape[0],
        ))

    def _predict_by_feat_single(
        self,
        ct_heatmap_pred: Tensor, 
        wh_pred: Tensor, 
        angle_pred: Tensor, 
        offset_pred: Tensor,
        img_meta: dict,
        instances: InstanceList,
        final: bool,
    ) -> InstanceData:
        """
        Reference mmdet.dense_heads.CenterNetHead._predict_by_feat_single().
        """
        batch_det_bboxes, batch_labels = self._decode_heatmap(
            ct_heatmap_pred[None],
            wh_pred[None],
            angle_pred[None],
            offset_pred[None],
            k=self.test_cfg.topk,
            kernel=self.test_cfg.local_maximum_kernel,
            score_threshold=self.test_cfg.score_threshold,
        )

        det_bboxes = batch_det_bboxes.view([-1, 6])
        det_labels = batch_labels.view(-1)
        if final:
            if instances:
                all_bboxes, all_labels = [det_bboxes], [det_labels]
                for instance in instances:
                    all_bboxes.append(torch.cat([instance.bboxes, instance.scores[..., None]], dim=-1))
                    all_labels.append(instance.labels)
                det_bboxes = torch.cat(all_bboxes)
                det_labels = torch.cat(all_labels)
                # Re-sort the results as the score order has been disrupted.
                sorted_idx = torch.argsort(det_bboxes[..., -1], descending=True)
                det_bboxes = det_bboxes[sorted_idx]
                det_labels = det_labels[sorted_idx]

            if self.test_cfg.with_nms:
                det_bboxes, det_labels = self._bboxes_nms(
                    det_bboxes, 
                    det_labels,
                    nms=self.test_cfg.nms,
                )
            if self.test_cfg.max_num > 0:
                det_bboxes = det_bboxes[:self.test_cfg.max_num]
                det_labels = det_labels[:self.test_cfg.max_num]

        results = InstanceData()
        results.bboxes = det_bboxes[..., :-1]
        results.scores = det_bboxes[..., -1]
        results.labels = det_labels
        return results
        
    def _decode_heatmap(
        self,
        ct_heatmap_pred: Tensor,
        wh_pred: Tensor,
        angle_pred: Tensor,
        offset_pred: Tensor,
        k: int = 100,
        kernel: int = 3,
        score_threshold: float = 0.,
    ) -> Tuple[Tensor, Tensor]:
        """
        Transform outputs into detections raw bbox prediction.

        Args:
            center_heatmap_pred (Tensor): center predict heatmap,
               shape (B, num_classes, H, W).
            wh_pred (Tensor): wh predict, shape (B, 2, H, W).
            angle_pred (Tensor): angle predict, shape (B, 1, H, W).
            offset_pred (Tensor): offset predict, shape (B, 2, H, W).
            k (int): Get top k center keypoints from heatmap. Defaults to 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
               Defaults to 3.

        Returns:
            tuple[Tensor]: Decoded output of CenterNetHead, containing
               the following Tensors:

              - batch_bboxes (Tensor): Coords of each box with shape (B, k, 5)
              - batch_topk_labels (Tensor): Categories of each box with \
                  shape (B, k)
        """
        ct_heatmap_pred = get_local_maximum(
            ct_heatmap_pred, kernel=kernel
        )

        topk_scores, topk_inds, topk_labels, topk_ys, topk_xs = self.get_topk_from_heatmap(
            ct_heatmap_pred, k=k, score_threshold=score_threshold
        )
        
        wh = transpose_and_gather_feat(wh_pred, topk_inds)
        angle = transpose_and_gather_feat(angle_pred, topk_inds)
        offset = transpose_and_gather_feat(offset_pred, topk_inds)

        topk_xs = topk_xs + offset[..., 0]
        topk_ys = topk_ys + offset[..., 1]
        topk_xs, topk_ys = SphereImageUtils.pxpy2theta_phi(
            [topk_xs, topk_ys], 
            w=self.erp_shape[1], 
            h=self.erp_shape[0]
        )
        
        batch_bboxes = torch.stack(
            [topk_xs, topk_ys, wh[..., 0], wh[..., 1], angle[..., 0]], 
            dim=2
        )
        batch_bboxes = torch.cat(
            (batch_bboxes, topk_scores[..., None]),
            dim=-1
        )
        return batch_bboxes, topk_labels

    def _bboxes_nms(
        self, 
        bboxes: Tensor,
        labels: Tensor,
        nms: dict,
    ) -> Tuple[Tensor, Tensor]:
        """bboxes nms."""
        if labels.numel() > 0:
            bboxes, keep = batched_nms(
                bboxes[:, :-1], 
                bboxes[:,-1].contiguous(),
                labels, 
                nms_cfg=nms
            )
            labels = labels[keep]

        return bboxes, labels

    @staticmethod
    def get_topk_from_heatmap(scores, k=20, score_threshold=0):
        """Get top k positions from heatmap.

        Args:
            scores (Tensor): Target heatmap with shape
                [batch, num_classes, height, width].
            k (int): Target number. Default: 20.

        Returns:
            tuple[torch.Tensor]: Scores, indexes, categories and coords of
                topk keypoint. Containing following Tensors:

            - topk_scores (Tensor): Max scores of each topk keypoint.
            - topk_inds (Tensor): Indexes of each topk keypoint.
            - topk_clses (Tensor): Categories of each topk keypoint.
            - topk_ys (Tensor): Y-coord of each topk keypoint.
            - topk_xs (Tensor): X-coord of each topk keypoint.
        """
        batch, _, height, width = scores.size()
        topk_scores, topk_inds = torch.topk(scores.view(batch, -1), k)
        if score_threshold > 0:
            mask = topk_scores > score_threshold
            topk_scores, topk_inds = topk_scores[mask], topk_inds[mask]
        topk_clses = torch.div(topk_inds, width * height, rounding_mode='trunc')
        topk_inds = topk_inds % (height * width)
        topk_ys = torch.div(topk_inds, width, rounding_mode='trunc')
        topk_xs = (topk_inds % width).int().float()
        return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs
