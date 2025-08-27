"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-01-28
@desc:   
"""
from torch import Tensor
import torch
from panorama_coco.sdk.r_centernet.calculate_RoIoU import Sph
from panorama_coco.utils.spherical import SphereImageUtils


class UnbiasedIoU:
    """ 
    Unbiased IoU Computation for Spherical Rectangles.
    bbox: [center_x, center_y, fov_x, fov_y, angle]
        center_x : [-pi, pi]
        center_y : [pi/2, -pi/2]
        fov_x    : [0, pi]
        fov_y    : [0, pi]
        angle    : [-pi/2, pi/2]
    """
    @classmethod                                               
    def cross_iou(cls, bboxes1: Tensor, bboxes2: Tensor) -> Tensor:
        """
        Unbiased Spherical IoU Computation.
        Args:
            bboxes1 (Tensor): It has shape (N, 5).
            bboxes2 (Tensor): It has shape (M, 5).

        Returns:
            Tensor: It has shape (N, M).
        """
        N, M = bboxes1.shape[0], bboxes2.shape[0]
        if N * M == 0:
            return torch.tensor([])
        bboxes1 = bboxes1.repeat_interleave(M, dim=0)
        bboxes2 = bboxes2.tile(N, 1)
        return cls.iou(bboxes1, bboxes2).view(N, M)
    
    @classmethod
    def iou(cls, bboxes1: Tensor, bboxes2: Tensor) -> Tensor:
        """ 
        Unbiased Spherical IoU Computation
        Unbiased Spherical IoU Computation.
        Args:
            x_bboxes (Tensor): It has shape (N, 5).
            y_bboxes (Tensor): It has shape (N, 5).

        Returns:
            Tensor: It has shape (N, 1).
        """
        assert bboxes1.shape == bboxes2.shape
        if bboxes1.shape[0] == 0:
            return torch.tensor([])
        
        bboxes1 = bboxes1.clone().detach()
        bboxes2 = bboxes2.clone().detach()
        bboxes1 = cls.transform(bboxes1)
        bboxes2 = cls.transform(bboxes2)
        
        return torch.from_numpy(
            Sph().sphIoU(bboxes1.cpu().numpy(), bboxes2.cpu().numpy())
        ).to(bboxes1.device)
    
    @staticmethod
    def transform(bboxes: Tensor) -> Tensor:
        """
        Change the format and range of the RBFoV Representations.
        Input:
        - gt: the last dimension: [center_x, center_y, fov_x, fov_y, angle]
            center_x : [-pi, pi]
            center_y : [pi/2, -pi/2]
            fov_x    : [0, pi]
            fov_y    : [0, pi]
            angle    : [-pi/2, pi/2]
            All parameters are angles.
        Output:
        - ann: the last dimension: [center_x', center_y', fov_x', fov_y', angle]
            center_x : [0, 2 * pi]
            center_y : [0, pi]
            fov_x    : [0, pi]
            fov_y    : [0, pi]
            angle    : [-pi/2, pi/2]
            All parameters are radians.
        """   
        bboxes[..., 0] = bboxes[..., 0] + torch.pi
        bboxes[..., 1] = torch.pi / 2 - bboxes[..., 1]
        return bboxes
    