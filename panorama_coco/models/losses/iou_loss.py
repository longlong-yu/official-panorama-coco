"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-03-26
@desc:   
"""
from mmrotate.models import RotatedIoULoss
import torch
from torch import Tensor
from panorama_coco.registry import MODELS


@MODELS.register_module()
class TangentIouLoss(RotatedIoULoss):
    """
    Tangent IOU loss.

    Args:
    """

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        *args,
        **kwargs
    ):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction (w, h, angle).
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".
        """
        new_pred = pred.new_zeros(pred.shape[:-1] + (5,))
        new_pred[..., 2] = 2 * torch.tan(pred[..., 0] / 2)
        new_pred[..., 3] = 2 * torch.tan(pred[..., 1] / 2)
        new_pred[..., 4] = pred[..., 2]
        
        new_target = torch.zeros_like(new_pred)
        new_target[..., 2] = 2 * torch.tan(target[..., 0] / 2)
        new_target[..., 3] = 2 * torch.tan(target[..., 1] / 2)
        new_target[..., 4] = target[..., 2]
        return super().forward(new_pred, new_target, *args, **kwargs)
