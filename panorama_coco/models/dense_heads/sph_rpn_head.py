"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-08-06
@desc:   Refer to https://github.com/AntXinyuan/sph2pob for the original implementation.
"""
from typing import List
from torch import Tensor
from mmdet.models.dense_heads import RPNHead
from mmdet.utils import InstanceList, OptInstanceList
from panorama_coco.models.dense_heads.base_anchor_head import AnchorHead
from panorama_coco.registry import MODELS


@MODELS.register_module()
class SphRPNHead(AnchorHead, RPNHead):

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    )-> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                has shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            batch_gt_instances (list[obj:InstanceData]): Batch of gt_instance.
                It usually includes ``bboxes`` and ``labels`` attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[obj:InstanceData], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        losses = super().loss_by_feat(
            cls_scores,
            bbox_preds,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore
        )
        return dict(
            loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox']
        )
