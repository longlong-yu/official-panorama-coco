"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-08-09
@desc:   
"""
from typing import List, Optional, Tuple
from mmdet.models.dense_heads import AnchorHead as MM_AnchorHead
from mmdet.models.utils import filter_scores_and_topk, images_to_levels, multi_apply, get_local_maximum
from mmdet.structures import SampleList
from mmdet.structures.bbox import cat_boxes, get_box_tensor
from mmdet.utils import InstanceList, OptInstanceList
from mmengine.config import ConfigDict
from mmengine.visualization import Visualizer
from torch import Tensor
import torch
from panorama_coco.core.nms import sph2pob_nms    # noqa
from panorama_coco.utils.spherical import SphereImageUtils
from panorama_coco.visualization.const import SampleItemType


class AnchorHead(MM_AnchorHead):
    """
    New Features & Updates:

    1. Added support for erp_weights.
    2. Introduced local_maximum_kernel configuration for post-processing.
    3. Enabled intermediate result visualization for debugging and analysis.
    4. Modified the behavior of the predict_rescale parameter.
    """
    
    def __init__(
        self, 
        *args, 
        with_erp_weights: bool = False, # Whether to multiply the predictions by erp_weights.
        predict_rescale: bool = True,   # Whether to perform rescaling during prediction.
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.with_erp_weights = with_erp_weights
        if self.test_cfg.with_nms and self.test_cfg.need_eval:
            self.test_cfg.nms.type = eval(self.test_cfg.nms.type)        
        self._erp_weights = None
        
        self.predict_resclae = predict_rescale
         
    def get_erp_weights(self, device):
        """ Get label weights — must initialize after retrieving the device at runtime. """
        if self._erp_weights is None:
            self._erp_weights = []        
            erp_h, erp_w = 512, 1024
            anchors_n = len(self.prior_generator.ratios) * len(self.prior_generator.scales)
            for stride_h, stride_w in self.prior_generator.strides:
                self._erp_weights.append(SphereImageUtils.erp_weights(
                    device=device,
                    erp_h=erp_h // stride_h,
                    erp_w=erp_w // stride_w,
                )[..., None].expand(-1, -1, anchors_n).reshape(-1) * 100)
        return self._erp_weights
    
    def predict(
        self,
        x: Tuple[Tensor],
        batch_data_samples: SampleList,
        rescale: bool = False,
        **kwargs,
    ) -> InstanceList:
        
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        outs = self(x)
        return self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=self.predict_resclae, **kwargs
        )
    
    def predict_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        score_factors: Optional[List[Tensor]] = None,
        batch_img_metas: Optional[List[dict]] = None,
        cfg: Optional[ConfigDict] = None,
        rescale: bool = False,
        with_nms: bool = True,
        *,
        instances_list: List[InstanceList] = None,
        final: bool = True,
        **kwargs
    ) -> InstanceList:
        # Added max_kernel_suppress.
        if self.test_cfg.get('local_maximum_kernel', 0) > 1:
            for level, cls_score in enumerate(cls_scores):
                for i in range(len(cls_score)):
                    if self.use_sigmoid_cls:
                        score = cls_score[i].sigmoid()
                    else:
                        # remind that we set FG labels to [0, num_class-1]
                        # since mmdet v2.0
                        # BG cat_id: num_class
                        score = cls_score[i].softmax(-1)[:, :-1]
                    score = get_local_maximum(
                        score, kernel=self.test_cfg.local_maximum_kernel
                    )
                    cls_score[i][score == 0.] = -9.21
                    

        return super().predict_by_feat(
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            score_factors=score_factors,
            batch_img_metas=batch_img_metas,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms
        )
     
    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                has shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device
        )
        
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore
        )
        (
            labels_list, label_weights_list, bbox_targets_list,
            bbox_weights_list, avg_factor
        ) = cls_reg_targets

        # ERP weights are used to mitigate the issue of excessive duplicate anchor boxes near the poles.
        if self.with_erp_weights:
            erp_label_weights = self.get_erp_weights(label_weights_list[0].device)
            for i in range(len(erp_label_weights)):
                # label_weights_list[i] *= erp_label_weights[i]
                bbox_weights_list[i] *= erp_label_weights[i][..., None].expand(-1, bbox_weights_list[i].shape[-1])
                 
        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(cat_boxes(anchor_list[i]))
        all_anchor_list = images_to_levels(
            concat_anchor_list,
            num_level_anchors
        )
        
        losses_cls, losses_bbox = multi_apply(
            self.loss_by_feat_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            avg_factor=avg_factor
        )
        
        # draw pred bboxes
        self._draw_feats(
            labels_list=labels_list,
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            all_anchor_list=all_anchor_list,
            bbox_targets_list=bbox_targets_list,
        )
         
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def _draw_feats(
        self,
        labels_list: List[Tensor],
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        all_anchor_list: List[Tensor],
        bbox_targets_list: List[Tensor] 
    ):
        # draw pred bboxes
        visualizer = Visualizer.get_current_instance()
        if not visualizer.check_mode():
            return
        
        channel_config = visualizer.mode_config.channel_configs[visualizer.MAIN_CHANNEL]
        batch_size = labels_list[0].shape[0]
        level_n = len(labels_list)
        for image_idx in range(batch_size):
            image_scores = []
            image_preds = []
            image_anchors = []
            image_labels = []
            image_targets = []
            for level_idx in range(level_n):
                level_scores = cls_scores[level_idx][image_idx]
                feat_h, feat_w = level_scores.shape[-2:]
                    
                level_scores = level_scores.sigmoid()
                level_scores = level_scores.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
                image_scores.append(level_scores)
                
                level_preds = bbox_preds[level_idx][image_idx]
                level_preds = level_preds.permute(1, 2, 0).reshape(-1, self.bbox_coder.encode_size)
                
                # (H, W, 9)
                level_anchors = all_anchor_list[level_idx][image_idx]
                level_anchors = level_anchors.reshape(-1, level_anchors.size(-1))
                level_preds = self.bbox_coder.decode(level_anchors, level_preds)
                level_preds = get_box_tensor(level_preds)
                image_anchors.append(get_box_tensor(level_anchors))
                image_preds.append(level_preds)
                
                level_labels = labels_list[level_idx][image_idx].reshape(-1)
                image_labels.append(level_labels)
                
                level_targets = bbox_targets_list[level_idx][image_idx]
                level_targets = level_targets.reshape(-1, level_targets.shape[-1])
                if not self.reg_decoded_bbox:
                    level_targets = self.bbox_coder.decode(level_anchors, level_targets)
                    level_targets = get_box_tensor(level_targets)
                image_targets.append(level_targets)
                    
                # H, W, anchor_n
                level_gt_heatmap = self.num_classes - level_labels.reshape(feat_h, feat_w, -1).permute(2, 0, 1)
                level_gt_heatmap = torch.where(level_gt_heatmap > 0, 1., 0).float()
                # heatmap gt
                visualizer.add_channel_item(
                    name=f'gt_heatmap_{level_idx}', 
                    value=level_gt_heatmap.detach(), 
                    item_type=SampleItemType.HAETMAP,
                )
                # heatmap pred
                level_pred_heatmap = torch.zeros_like(level_gt_heatmap)
                for label_idx in range(len(level_labels)):
                    if label_idx < self.num_classes:
                        level_pred_heatmap += level_scores[..., label_idx].reshape(feat_h, feat_w, -1).permute(2, 0, 1)
                visualizer.add_channel_item(
                    name=f'pred_heatmap_{level_idx}', 
                    value=level_pred_heatmap.detach(), 
                    item_type=SampleItemType.HAETMAP,
                )
                
                # level pred 
                results = filter_scores_and_topk(
                    level_scores, 
                    channel_config.score_thr,
                    channel_config.max_bbox_n,
                    dict(bbox_pred=level_preds)
                )
                level_scores_, level_image_pred_labels, _, filtered_results = results

                level_preds_ = filtered_results['bbox_pred']
                visualizer.add_channel_item(
                    name=f'pred_sbox_{level_idx}', 
                    value=level_preds_.detach(),
                    scores=level_scores_.detach(),
                    labels=level_image_pred_labels,
                    item_type=SampleItemType.SBOX,
                )
                    
                # level bboxes gt
                level_labels = self.num_classes - level_labels.reshape(feat_h, feat_w, -1).permute(2, 0, 1).float()
                level_labels += torch.rand_like(level_labels) / 10
                level_labels[:] = get_local_maximum(
                    level_labels[:], kernel=channel_config.local_maximum_kernel
                ).int()
                level_labels = self.num_classes - level_labels 
                level_labels = level_labels.permute(1, 2, 0).reshape(-1)
                target_idx = level_labels < self.num_classes
                level_targets = level_targets[target_idx]
                tmp = torch.cat([
                        level_targets * 180 / torch.pi,
                        level_labels[target_idx, None]
                    ],
                    dim=-1
                ).unique(dim=0)
                visualizer.add_channel_item(
                    name=f'gt_sbox_{level_idx}', 
                    value=tmp[..., :-1].detach() * torch.pi / 180,
                    labels=tmp[..., -1].int().detach(),
                    item_type=SampleItemType.SBOX,
                ) 
                
            image_scores = torch.cat(image_scores)
            image_preds = torch.cat(image_preds)
            image_anchors = torch.cat(image_anchors)
            image_labels = torch.cat(image_labels)
            image_targets = torch.cat(image_targets)  
            
            # pred 
            results = filter_scores_and_topk(
                image_scores, 
                channel_config.score_thr,
                channel_config.max_bbox_n,
                dict(bbox_pred=image_preds)
            )
            image_scores_, image_pred_labels, _, filtered_results = results

            image_preds_ = filtered_results['bbox_pred']
            visualizer.add_channel_item(
                name=f'pred_sbox_all', 
                value=image_preds_.detach(),
                scores=image_scores_.detach(),
                labels=image_pred_labels,
                item_type=SampleItemType.SBOX,
            )
            
            # bboxes gt
            target_idx = image_labels < self.num_classes 
            image_targets = image_targets[target_idx]
            tmp = torch.cat([
                    (image_targets * 180 / torch.pi).int(),
                    image_labels[target_idx, None]
                ],
                dim=-1
            ).unique(dim=0)
            visualizer.add_channel_item(
                name=f'gt_sbox_all', 
                value=tmp[..., :-1].detach() * torch.pi / 180,
                labels=tmp[..., -1].int().detach(),
                item_type=SampleItemType.SBOX,
            )
