import mmcv
import numpy as np
import torch

from mmdet.models.task_modules import BaseBBoxCoder
from mmdet.structures.bbox import get_box_type
from panorama_coco.registry import TASK_UTILS
from panorama_coco.utils.norm import normalize
from panorama_coco.utils.spherical import FLOAT_EPS


ANGLE_MODE_L1 = 'l1'
ANGLE_MODE_SIN = 'sin'


@TASK_UTILS.register_module()
class DeltaXYWHASphBBoxCoder(BaseBBoxCoder):
    """Delta XYWH BBox coder.

    Following the practice in `R-CNN <https://arxiv.org/abs/1311.2524>`_,
    this coder encodes bbox (theta, phi, alpha, beta) into delta (d_theta, d_phi, d_alpha, d_beta) and
    decodes delta (d_theta, d_phi, d_alpha, d_beta) back to original bbox (theta, phi, alpha, beta).

    Args:
        target_means (Sequence[float]): Denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): Denormalizing standard deviation of
            target for delta coordinates
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.
        add_ctr_clamp (bool): Whether to add center clamp, when added, the
            predicted box is clamped is its center is too far away from
            the original anchor's center. Only used by YOLOF. Default False.
        ctr_clamp (int): the maximum pixel shift to clamp. Only used by YOLOF.
            Default 32.
    """
    encode_size = 5
    
    def __init__(
        self,
        target_means=(0., 0., 0., 0., 0.),
        target_stds=(1., 1., 1., 1., 1.),
        clip_border=True,
        add_ctr_clamp=False,
        ctr_clamp=32,
        box_type: str = None,
        delta_ratio: float = 1.0, # !Note: should only be used with ANGLE_MODE_L1 
        angle_mode: str = ANGLE_MODE_L1
    ):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds
        self.clip_border = clip_border
        self.add_ctr_clamp = add_ctr_clamp
        self.ctr_clamp = ctr_clamp
        self.box_type = box_type
        self.delta_ratio = delta_ratio
        self.angle_mode = angle_mode

    def encode(self, bboxes, gt_bboxes):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (torch.Tensor): Source boxes, e.g., object proposals.
            gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """

        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 5
        return bbox2delta(
            bboxes, gt_bboxes, self.means, self.stds, self.delta_ratio, self.angle_mode
        )


    def decode(
        self,
        bboxes,
        pred_bboxes,
        max_shape=None,
        wh_ratio_clip=16 / 1000
    ):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            bboxes (torch.Tensor): Basic boxes. Shape (B, N, 4) or (N, 4)
            pred_bboxes (Tensor): Encoded offsets with respect to each roi.
               Has shape (B, N, num_classes * 4) or (B, N, 4) or
               (N, num_classes * 4) or (N, 4). Note N = num_anchors * W * H
               when rois is a grid of anchors.Offset encoding follows [1]_.
            max_shape (Sequence[int] or torch.Tensor or Sequence[
               Sequence[int]],optional): Maximum bounds for boxes, specifies
               (H, W, C) or (H, W). If bboxes shape is (B, N, 4), then
               the max_shape should be a Sequence[Sequence[int]]
               and the length of max_shape should also be B.
            wh_ratio_clip (float, optional): The allowed ratio between
                width and height.

        Returns:
            torch.Tensor: Decoded boxes.
        """

        assert pred_bboxes.size(0) == bboxes.size(0)
        if pred_bboxes.ndim == 3:
            assert pred_bboxes.size(1) == bboxes.size(1)

        if pred_bboxes.ndim == 2 and not torch.onnx.is_in_onnx_export():
            # single image decode
            decoded_bboxes = delta2bbox(
                bboxes, pred_bboxes, self.means,
                self.stds, max_shape, wh_ratio_clip,
                self.clip_border, self.add_ctr_clamp,
                self.ctr_clamp, self.delta_ratio,
                self.angle_mode
            )
        else:
            raise NotImplemented('omnx function is not implement!')
        
        if self.box_type is not None:
            _, cls_type = get_box_type(self.box_type)
            decoded_bboxes = cls_type(decoded_bboxes)
            
        return decoded_bboxes


@mmcv.utils.jit(coderize=True)
def bbox2delta(
    proposals, 
    gt, 
    means=(0., 0., 0., 0., 0.),
    stds=(1., 1., 1., 1., 1.),
    delta_ratio: float = 1.0,
    angle_mode: str = ANGLE_MODE_L1
):
    """Compute deltas of proposals w.r.t. gt.

    We usually compute the deltas of x, y, w, h of proposals w.r.t ground
    truth bboxes to get regression target.
    This is the inverse function of :func:`delta2bbox`.

    Args:
        proposals (Tensor): Boxes to be transformed, shape (N, ..., 4)
        gt (Tensor): Gt bboxes to be used as base, shape (N, ..., 4)
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates

    Returns:
        Tensor: deltas with shape (N, 4), where columns represent dx, dy,
            dw, dh.
    """
    assert proposals.size() == gt.size()

    if not torch.torch.is_tensor(gt):
        gt = gt.tensor

    eps = 1e-7
    if not torch.is_tensor(proposals):
        proposals = proposals.tensor
    proposals = proposals.float() * delta_ratio
    gt = gt.float() * delta_ratio
    px = proposals[..., 0]
    py = proposals[..., 1]
    pw = proposals[..., 2].clip(min=eps)
    ph = proposals[..., 3].clip(min=eps)
    pa = proposals[..., 4]

    gx = gt[..., 0]
    gy = gt[..., 1]
    gw = gt[..., 2].clip(min=eps)
    gh = gt[..., 3].clip(min=eps)
    ga = gt[..., 4]

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    if angle_mode == ANGLE_MODE_SIN:
        da = torch.sin(ga - pa)
    else:
        da = ga - pa
    da = normalize(da, min_=-torch.pi / 2, max_=torch.pi / 2)
    deltas = torch.stack([dx, dy, dw, dh, da], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


@mmcv.utils.jit(coderize=True)
def delta2bbox(
    rois,
    deltas,
    means=(0., 0., 0., 0., 0.),
    stds=(1., 1., 1., 1., 1.),
    max_shape=None,
    wh_ratio_clip: float = 16 / 1000,
    clip_border: bool = True,
    add_ctr_clamp: bool = False,
    ctr_clamp: int = 32,
    delta_ratio: float = 1.0,
    angle_mode: str = ANGLE_MODE_L1
):
    """Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.
    This is the inverse function of :func:`bbox2delta`.

    Args:
        rois (Tensor): Boxes to be transformed. Has shape (N, 4).
        deltas (Tensor): Encoded offsets relative to each roi.
            Has shape (N, num_classes * 4) or (N, 4). Note
            N = num_base_anchors * W * H, when rois is a grid of
            anchors. Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates.
            Default (0., 0., 0., 0.).
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates. Default (1., 1., 1., 1.).
        max_shape (tuple[int, int]): Maximum bounds for boxes, specifies
           (H, W). Default None.
        wh_ratio_clip (float): Maximum aspect ratio for boxes. Default
            16 / 1000.
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Default True.
        add_ctr_clamp (bool): Whether to add center clamp. When set to True,
            the center of the prediction bounding box will be clamped to
            avoid being too far away from the center of the anchor.
            Only used by YOLOF. Default False.
        ctr_clamp (int): the maximum pixel shift to clamp. Only used by YOLOF.
            Default 32.

    Returns:
        Tensor: Boxes with shape (N, num_classes * 4) or (N, 4), where 4
           represent tl_x, tl_y, br_x, br_y.

    References:
        .. [1] https://arxiv.org/abs/1311.2524

    Example:
        >>> rois = torch.Tensor([[ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 5.,  5.,  5.,  5.]])
        >>> deltas = torch.Tensor([[  0.,   0.,   0.,   0.],
        >>>                        [  1.,   1.,   1.,   1.],l
        >>>                        [  0.,   0.,   2.,  -1.],
        >>>                        [ 0.7, -1.9, -0.5,  0.3]])
        >>> delta2bbox(rois, deltas, max_shape=(32, 32, 3))
        tensor([[0.0000, 0.0000, 1.0000, 1.0000],
                [0.1409, 0.1409, 2.8591, 2.8591],
                [0.0000, 0.3161, 4.1945, 0.6839],
                [5.0000, 5.0000, 5.0000, 5.0000]])
    """
    num_bboxes, num_classes = deltas.size(0), deltas.size(1) // 5
    if num_bboxes == 0:
        return deltas
    
    deltas = deltas.reshape(-1, 5)

    means = deltas.new_tensor(means).view(1, -1)
    stds = deltas.new_tensor(stds).view(1, -1)
    denorm_deltas = deltas * stds + means

    dxy = denorm_deltas[:, :2]
    dwh = denorm_deltas[:, 2:4]
    da  = denorm_deltas[:, 4]
    if angle_mode == ANGLE_MODE_SIN:
        da = torch.arcsin(da.clamp_(min=-1 + FLOAT_EPS, max=1 - FLOAT_EPS))

    # Compute width/height of each roi
    if not torch.is_tensor(rois):
        rois = rois.tensor
        
    # Increase the scaling factor to improve the resolution of delta values.
    rois = rois * delta_ratio

    rois = rois.repeat(1, num_classes).reshape(-1, 5) 
    pxy = rois[:, :2]
    pwh = rois[:, 2:4]
    pa  = rois[:, 4]

    dxy_wh = pwh * dxy

    max_ratio = np.abs(np.log(wh_ratio_clip))
    if add_ctr_clamp:
        dxy_wh = torch.clamp_(dxy_wh, max=ctr_clamp, min=-ctr_clamp)
        dwh = torch.clamp_(dwh, max=max_ratio)
    else:
        dwh = dwh.clamp_(min=-max_ratio, max=max_ratio)

    if clip_border:
        dxy_wh = dxy_wh.clamp_(min=-torch.pi / 2 * delta_ratio, max=torch.pi / 2 * delta_ratio)
        da = da.clamp_(min=-torch.pi / 2 * delta_ratio, max=torch.pi / 2 * delta_ratio)
        
    gxy = pxy + dxy_wh
    gwh = pwh * dwh.exp()
    ga  = pa + da
    bboxes = torch.cat([gxy, gwh, ga[:, None]], dim=-1)
    
    # Rescale to actual size.
    bboxes = bboxes / delta_ratio
    
    if clip_border:
        eps = 1e-7
        bboxes[..., 0] = normalize(bboxes[..., 0], min_=-torch.pi, max_=torch.pi, eps=eps)
        bboxes[..., 1] = normalize(bboxes[..., 1], min_=-torch.pi / 2, max_=torch.pi / 2, eps=eps)
        bboxes[..., 2:4].clamp_(min=eps, max=torch.pi - eps)
        bboxes[..., 4] = normalize(bboxes[..., 4], min_=-torch.pi / 2, max_=torch.pi / 2, eps=eps)
            
    return bboxes
