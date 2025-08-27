from typing import List, Optional, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from mmengine.visualization import Visualizer
from mmdet.models.layers import multiclass_nms
from mmdet.models.losses import accuracy
from mmdet.models.roi_heads import Shared2FCBBoxHead, StandardRoIHead
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.models.utils import empty_instances, filter_scores_and_topk
from mmdet.structures.bbox import bbox2roi, get_box_tensor, get_box_type
from mmdet.utils import ConfigType, InstanceList

from panorama_coco.core.nms import sph2pob_nms # noqa
from panorama_coco.registry import MODELS
from panorama_coco.utils.bbox import _sph2pix_box_transform, r2hbox_xywh
from panorama_coco.utils.spherical import FLOAT_EPS, ImageUtils, SphereImageUtils
from panorama_coco.visualization.const import SampleItemType


@MODELS.register_module()
class SphShared2FCBBoxHead(Shared2FCBBoxHead):

    def __init__(
        self,
        *args,
        reg_predictor_cfg: ConfigType = dict(type='mmdet.Linear'),
        cls_predictor_cfg: ConfigType = dict(type='mmdet.Linear'),
        **kwargs,
    ):
        super().__init__(
            *args,
            reg_predictor_cfg=reg_predictor_cfg,
            cls_predictor_cfg=cls_predictor_cfg,
            **kwargs,
        )
    
    def _predict_by_feat_single(
            self,
            roi: Tensor,
            cls_score: Tensor,
            bbox_pred: Tensor,
            img_meta: dict,
            rescale: bool = False,
            rcnn_test_cfg: Optional[ConfigDict] = None) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            roi (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_meta (dict): image information.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head.
                Defaults to None

        Returns:
            :obj:`InstanceData`: Detection results of each image\
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, ) 
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        results = InstanceData()
        if roi.shape[0] == 0:
            return empty_instances([img_meta],
                                   roi.device,
                                   task_type='bbox',
                                   instance_results=[results],
                                   box_type=self.predict_box_type,
                                   use_box_type=False,
                                   num_classes=self.num_classes,
                                   score_per_cls=rcnn_test_cfg is None)[0]

        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None

        img_shape = img_meta['img_shape']
        num_rois = roi.size(0)
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            num_classes = 1 if self.reg_class_agnostic else self.num_classes
            roi = roi.repeat_interleave(num_classes, dim=0)
            bbox_pred = bbox_pred.view(-1, self.bbox_coder.encode_size)
            bboxes = self.bbox_coder.decode(
                roi[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = roi[:, 1:].clone()
            if img_shape is not None and bboxes.size(-1) == 4:
                # bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                # bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])
                eps = 1e-7
                bboxes[:, 2].clamp_(min=eps, max=torch.pi - eps)
                bboxes[:, 3].clamp_(min=eps, max=torch.pi - eps)

        # if rescale and bboxes.size(0) > 0:
        #     assert img_meta.get('scale_factor') is not None
        #     scale_factor = [1 / s for s in img_meta['scale_factor']]
        #     bboxes = scale_boxes(bboxes, scale_factor)

        # Get the inside tensor when `bboxes` is a box type
        bboxes = get_box_tensor(bboxes)
        box_dim = bboxes.size(-1)
        bboxes = bboxes.view(num_rois, -1)

        if rcnn_test_cfg is None:
            # This means that it is aug test.
            # It needs to return the raw results without nms.
            results.bboxes = bboxes
            results.scores = scores
        else:
            det_bboxes, det_labels = multiclass_nms(
                bboxes,
                scores,
                rcnn_test_cfg.score_thr,
                rcnn_test_cfg.nms,
                rcnn_test_cfg.max_per_img,
                box_dim=box_dim)
            results.bboxes = det_bboxes[:, :-1]
            results.scores = det_bboxes[:, -1]
            results.labels = det_labels
        return results

    def loss_and_target(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        rois: Tensor,
        sampling_results: List[SamplingResult],
        rcnn_train_cfg: ConfigDict,
        concat: bool = True,
        reduction_override: Optional[str] = None
    ) -> dict:
        """Calculate the loss based on the features extracted by the bbox head.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            rois (Tensor): RoIs with the shape
                (batch_size * num_proposals_single_image, 5) where the first
                column indicates batch id of each RoI.
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch. Defaults to True.
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

        Returns:
            dict: A dictionary of loss and targets components.
                The targets are only used for cascade rcnn.
        """

        cls_reg_targets = self.get_targets(
            sampling_results, rcnn_train_cfg, concat=concat
        )
        losses = self.loss(
            cls_score,
            bbox_pred,
            rois,
            *cls_reg_targets,
            reduction_override=reduction_override,
            sampling_results=sampling_results, 
        )

        # cls_reg_targets is only for cascade rcnn
        return dict(loss_bbox=losses, bbox_targets=cls_reg_targets)
    
    def loss(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        rois: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        bbox_targets: Tensor,
        bbox_weights: Tensor,
        sampling_results: List[SamplingResult],
        reduction_override: Optional[str] = None,
    ) -> dict:
        """Calculate the loss based on the network predictions and targets.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            rois (Tensor): RoIs with the shape
                (batch_size * num_proposals_single_image, 5) where the first
                column indicates batch id of each RoI.
            labels (Tensor): Gt_labels for all proposals in a batch, has
                shape (batch_size * num_proposals_single_image, ).
            label_weights (Tensor): Labels_weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, ).
            bbox_targets (Tensor): Regression target for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4),
                the last dimension 4 represents [tl_x, tl_y, br_x, br_y].
            bbox_weights (Tensor): Regression weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4).
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

        Returns:
            dict: A dictionary of loss.
        """

        losses = dict()

        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_num = bbox_pred.shape[0]
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                    bbox_pred = get_box_tensor(bbox_pred)
                    bbox_pred = bbox_pred.view(bbox_num, -1)

                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), self.num_classes,
                        -1)[pos_inds.type(torch.bool),
                            labels[pos_inds.type(torch.bool)]]

                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
                
        # draw
        self._draw_feats(
            rois=rois,
            cls_scores=cls_score,
            bbox_preds=bbox_pred,
            pos_inds=pos_inds,
            labels=labels,
            bbox_targets=bbox_targets,
            sampling_results=sampling_results,
        )
        return losses
    
    def _draw_feats(
        self,
        rois: Tensor,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        pos_inds: Tensor,
        labels: Tensor,
        bbox_targets: Tensor,
        sampling_results: List[SamplingResult], 
    ):
        visualizer = Visualizer.get_current_instance()
        if not visualizer.check_mode():
            return
        
        if not self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`,
            # `GIouLoss`, `DIouLoss`) is applied directly on
            # the decoded bounding boxes, it decodes the
            # already encoded coordinates to absolute format.
            bbox_targets = self.bbox_coder.decode(rois[:, 1:], bbox_targets)
            bbox_targets = get_box_tensor(bbox_targets)
            bbox_preds = self.bbox_coder.decode(rois[:, 1:], bbox_preds)
            bbox_preds = get_box_tensor(bbox_preds).reshape(bbox_targets.shape[0], -1)
         
        channel_config = visualizer.mode_config.channel_configs[visualizer.MAIN_CHANNEL]
        num_proposals_per_img = tuple(len(p.priors) for p in sampling_results)
        rois = rois.split(num_proposals_per_img, 0)
        bbox_preds = bbox_preds.split(num_proposals_per_img, 0)
        pos_inds = pos_inds.split(num_proposals_per_img, 0)
        labels = labels.split(num_proposals_per_img, 0)
        bbox_targets = bbox_targets.split(num_proposals_per_img, 0)
        
        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            cls_scores = self.loss_cls.get_activation(cls_scores)
        else:
            cls_scores = F.softmax(cls_scores, dim=-1) if cls_scores is not None else None
        cls_scores = cls_scores[..., :-1].split(num_proposals_per_img, 0)
        
        for cls_score, bbox_pred, label, bbox_target, pos_ind in zip(
            cls_scores, bbox_preds, labels, bbox_targets, pos_inds
        ):  
            pos_ind = pos_ind.bool()
            bbox_pred = bbox_pred.view(bbox_pred.size(0), self.num_classes, -1)
            # !Note: Append rotation angle to bbox before computing the loss.
            if bbox_pred.shape[-1] == 4:
                bbox_pred = torch.cat([bbox_pred, torch.zeros_like(bbox_pred[..., [0]])], dim=-1) 
            results = filter_scores_and_topk(
                cls_score[pos_ind], 
                channel_config.score_thr,
                channel_config.max_bbox_n
            )
            cls_score, label_pred, keep_idxs, _ = results
            bbox_pred = bbox_pred[pos_ind][keep_idxs,label_pred]
            visualizer.add_channel_item(
                name=f'pred_sbox_roi',
                value=bbox_pred.detach(),
                scores=cls_score.detach(),
                labels=label_pred.detach(),
                item_type=SampleItemType.SBOX,
            )
            
            visualizer.add_channel_item(
                name=f'gt_sbox_roi', 
                value=bbox_target[pos_ind].detach(),
                labels=label[pos_ind].detach(),
                item_type=SampleItemType.SBOX,
            )  

@MODELS.register_module()
class SphStandardRoIHead(StandardRoIHead):
    def __init__(self, image_shape: Tuple[int, int], *args, **kwargs):
        self.image_shape = image_shape
        super().__init__(*args, **kwargs)
        if self.test_cfg.need_eval:
            self.test_cfg.nms.type = eval(self.test_cfg.nms.type)

    def _formalize_rois(self, rois: Tensor) -> Tensor:
        """ Ensure rois bboxe type is consistent with delta decoder. """
        _, cls_type = get_box_type(self.bbox_head.bbox_coder.box_type)
        sboxes = cls_type(rois[:, 1:])
        return torch.cat([rois[:, [0]], get_box_tensor(sboxes)], dim=1)
        
    def _bbox_forward(self, x: Tuple[Tensor], rois: Tensor) -> dict:
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        img_shape = self.image_shape
         
        with torch.no_grad():
            rois_ = rois.clone()
            bboxes, h_alpha, h_beta = SphereImageUtils.sbox2erp(rois_[:, 1:], estimate=False)
            bboxes[..., 0:2] = SphereImageUtils.theta_phi2pxpy(
                theta_phi=bboxes[..., 0:2].permute(1, 0),
                w=img_shape[1],
                h=img_shape[0],
                clamp=False,
            ).permute(1, 0)
            bboxes[..., 2] = bboxes[..., 2] * img_shape[0] / torch.pi
            bboxes[..., 3] = bboxes[..., 3] * img_shape[1] / torch.pi / 2
            bboxes = ImageUtils.xywh2xyxy(bboxes[:, :4])
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
            rois_xyxy = torch.cat([rois_[:, [0]], bboxes], dim=1)
            
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois_xyxy, rois
        )
        
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)
        
        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred.reshape(bbox_pred.shape[0], -1, ), bbox_feats=bbox_feats
        )
        
        # draw feats
        self._draw_feats(rois, rois_xyxy)
        
        return bbox_results, h_alpha, h_beta
 
    def _draw_feats(
        self,
        rois: Tensor,
        rois_xyxy: Tensor,
    ):
        visualizer = Visualizer.get_current_instance()
        if not visualizer.check_mode():
            return
        
        # calculate shboxes
        sboxes = rois[..., 1:].clone()
        sboxes[..., 2:4] = sboxes[..., 2:4].clamp_(
            min=0 + 2 * FLOAT_EPS,
            max=torch.pi - 2 * FLOAT_EPS,
        )
        # tangent plane
        w = 2 * torch.tan(sboxes[..., 2] / 2)
        h = 2 * torch.tan(sboxes[..., 3] / 2)
        h_w = w * torch.cos(sboxes[..., 4]).abs() + h * torch.sin(sboxes[..., 4]).abs()
        h_h = w * torch.sin(sboxes[..., 4]).abs() + h * torch.cos(sboxes[..., 4]).abs()
        h_alpha = 2 * torch.atan(h_w / 2)
        h_beta = 2 * torch.atan(h_h / 2)
        sboxes[..., 2] = h_alpha
        sboxes[..., 3] = h_beta
        sboxes[..., 4] = 0
        
        image_num = rois_xyxy[..., 0].max().int()
        for i in range(image_num + 1):
            mask = rois_xyxy[..., 0] == i
            visualizer.add_channel_item(
                name=f'gt_roi_hbox_roi',
                value=rois_xyxy[mask][..., 1:],
                item_type=SampleItemType.HBOX,
            )
            
            visualizer.add_channel_item(
                name=f'gt_roi_sbox_roi',
                value=rois[mask][..., 1:], 
                item_type=SampleItemType.SBOX,
            )
            
            visualizer.add_channel_item(
                name=f'gt_roi_shbox_roi',
                value=sboxes[mask],
                item_type=SampleItemType.SBOX,
            )

    def bbox_loss(
        self, 
        x: Tuple[Tensor],
        sampling_results: List[SamplingResult]
    ) -> dict:
        """Perform forward propagation and loss calculation of the bbox head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
        """
        rois = bbox2roi([res.priors for res in sampling_results])
        bbox_results, h_alpha, h_beta = self._bbox_forward(x, rois)
        # use shbox instead of sbox
        rois = torch.cat((
            rois[..., 0:3], 
            h_alpha[..., None], 
            h_beta[..., None], 
            torch.zeros_like(rois[..., [0]])
        ), dim=-1)
        rois = self._formalize_rois(rois)

        bbox_loss_and_target = self.bbox_head.loss_and_target(
            cls_score=bbox_results['cls_score'],
            bbox_pred=bbox_results['bbox_pred'],
            rois=rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg)

        bbox_results.update(loss_bbox=bbox_loss_and_target['loss_bbox'])
        return bbox_results
    
    def predict_bbox(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     rpn_results_list: InstanceList,
                     rcnn_test_cfg: ConfigType,
                     rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the bbox head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        proposals = [res.bboxes for res in rpn_results_list]
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            return empty_instances(
                batch_img_metas,
                rois.device,
                task_type='bbox',
                box_type=self.bbox_head.predict_box_type,
                num_classes=self.bbox_head.num_classes,
                score_per_cls=rcnn_test_cfg is None)

        bbox_results, h_alpha, h_beta = self._bbox_forward(x, rois)
        # use shbox instead of sbox
        rois = torch.cat((
            rois[..., 0:3], 
            h_alpha[..., None], 
            h_beta[..., None], 
            torch.zeros_like(rois[..., [0]])
        ), dim=-1)
        rois = self._formalize_rois(rois)

        # split batch bbox prediction back to each image
        cls_scores = bbox_results['cls_score']
        bbox_preds = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_scores = cls_scores.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_preds will be None
        if bbox_preds is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_preds, torch.Tensor):
                bbox_preds = bbox_preds.split(num_proposals_per_img, 0)
            else:
                bbox_preds = self.bbox_head.bbox_pred_split(
                    bbox_preds, num_proposals_per_img)
        else:
            bbox_preds = (None, ) * len(proposals)

        result_list = self.bbox_head.predict_by_feat(
            rois=rois,
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=rcnn_test_cfg,
            rescale=rescale)
        return result_list
