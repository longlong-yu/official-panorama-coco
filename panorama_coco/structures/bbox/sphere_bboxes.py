"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-03-23
@desc:   
"""
from typing import Optional, Tuple
from mmdet.structures.bbox import register_box
from mmrotate.structures.bbox import RotatedBoxes
import numpy as np
from torch import Tensor
import torch

from panorama_coco.utils.norm import normalize
from panorama_coco.utils.spherical import SphereImageUtils


@register_box('sbox')
class SphereBoxes(RotatedBoxes):
    """The rotated box class used in MMRotate by default.

    The ``box_dim`` of ``RotatedBoxes`` is 5, which means the length of the
    last dimension of the input should be 5. Each row of data means
    (x, y, w, h, t), where 'x' and 'y' are the coordinates of the box center,
    'w' and 'h' are the length of box sides, 't' is the box angle represented
    in radian. A rotated box can be regarded as rotating the horizontal box
    (x, y, w, h) w.r.t its center by 't' radian CW.

    Args:
        data (Tensor or np.ndarray or Sequence): The box data with shape
            (..., 5).
        dtype (torch.dtype, Optional): data type of boxes. Defaults to None.
        device (str or torch.device, Optional): device of boxes.
            Default to None.
        clone (bool): Whether clone ``boxes`` or not. Defaults to True.
    """

    box_dim: int = 5

    def __init__(
        self, 
        *args, 
        with_norm: bool = True,
        with_regular: bool = True,
        clone: bool = True,
        **kwargs
    ) -> None:
        super().__init__(*args, clone=clone, **kwargs)
        # !Note: Skip norm and regular when reconstructing from sbox to avoid redundant copying.
        if clone:
            # Remove periodicity.
            if with_norm:
                bboxes = self.tensor
                bboxes[..., 0] = normalize(bboxes[..., 0], min_=-torch.pi, max_=torch.pi)
                bboxes[..., 1] = normalize(bboxes[..., 1], min_=-torch.pi / 2, max_=torch.pi / 2)
                bboxes[..., 4] = normalize(bboxes[..., 4], min_=-torch.pi / 2, max_=torch.pi / 2)
            if with_regular:
                self.regularize_boxes(pattern='le90')
    
    def regularize_boxes(
        self,
        pattern: Optional[str] = None,
        width_longer: bool = True,
        start_angle: float = -90
    ) -> Tensor:
        """
        Regularize rotated boxes. Refer to RotatedBoxes.regularize_boxes.
        Fixed the issue where each regularization would rotate 90 degrees when the length and width were equal.
        """
        boxes = self.tensor
        if pattern is not None:
            if pattern == 'oc':
                width_longer, start_angle = False, -90
            elif pattern == 'le90':
                width_longer, start_angle = True, -90
            elif pattern == 'le135':
                width_longer, start_angle = True, -45
            else:
                raise ValueError("pattern only can be 'oc', 'le90', and"
                                 f"'le135', but get {pattern}.")
        start_angle = start_angle / 180 * np.pi

        x, y, w, h, t = boxes.unbind(dim=-1)
        if width_longer:
            # swap edge and angle if h > w
            w_ = torch.where(w >= h, w, h)
            h_ = torch.where(w >= h, h, w)
            t = torch.where(w >= h, t, t + np.pi / 2)
            t = ((t - start_angle) % np.pi) + start_angle
        else:
            # swap edge and angle if angle > pi/2
            t = ((t - start_angle) % np.pi)
            w_ = torch.where(t < np.pi / 2, w, h)
            h_ = torch.where(t < np.pi / 2, h, w)
            t = torch.where(t < np.pi / 2, t, t - np.pi / 2) + start_angle
        self.tensor = torch.stack([x, y, w_, h_, t], dim=-1)
        return self.tensor
     
    @property
    def centers(self) -> Tensor:
        """Return a tensor representing the centers of boxes.

        If boxes have shape of (m, 8), centers have shape of (m, 2).
        """
        return self.tensor[..., :2]

    @property
    def areas(self) -> Tensor:
        """Return a tensor representing the areas of boxes.

        If boxes have shape of (m, 8), areas have shape of (m, ).
        """
        return SphereImageUtils.sbox_area(self.tensor)

    @property
    def widths(self) -> Tensor:
        """Return a tensor representing the widths of boxes.

        If boxes have shape of (m, 8), widths have shape of (m, ).
        """
        return self.tensor[..., 2]

    @property
    def heights(self) -> Tensor:
        """Return a tensor representing the heights of boxes.

        If boxes have shape of (m, 8), heights have shape of (m, ).
        """
        return self.tensor[..., 3]
    
    def flip_(self,
        img_shape: Tuple[int, int],
        direction: str = 'horizontal'
    ) -> None:
        """Flip boxes horizontally or vertically in-place.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
            direction (str): Flip direction, options are "horizontal",
                "vertical" and "diagonal". Defaults to "horizontal"
        """
        assert direction in ['horizontal', 'vertical', 'diagonal']
        flipped = self.tensor
        if direction == 'horizontal':
            flipped[..., 0] = -flipped[..., 0]
            flipped[..., 4] = -flipped[..., 4]
        elif direction == 'vertical':
            flipped[..., 1] = -flipped[..., 1]
            flipped[..., 4] = -flipped[..., 4]
        else:
            flipped[..., 0] = -flipped[..., 0]
            flipped[..., 1] = -flipped[..., 1]

    def rotate(
        self,
        roll: float, 
        pitch: float, 
        yaw: float,
        z_down: bool = False,
    ) -> None:
        """ Rotate all boxes in-place. """ 
        xyz = SphereImageUtils.theta_phi2xyz(self.tensor[..., 0:2].permute(1, 0))
        xyz_r = SphereImageUtils.rotate_xyz(
            xyz,
            roll=-pitch,
            pitch=-yaw,
            yaw=-roll,
            order=[1, 0, 2],
            z_down=False,
        )
        self.tensor[..., 0:2] = SphereImageUtils.xyz2theta_phi(xyz_r).permute(1, 0)
        os_v_r = xyz_r.permute(1, 0) 
        oz_v = os_v_r.new_tensor([0, 1.0, 0]).expand(os_v_r.shape[0], -1)
        
        n1 = torch.cross(os_v_r, oz_v)
        n1 = torch.nn.functional.normalize(n1, p=2, dim=-1)
        xyz_a = SphereImageUtils.rotate(xyz, oz_v.permute(1, 0), self.tensor[..., -1])
        oa_v_r = SphereImageUtils.rotate_xyz(
            xyz_a,
            roll=-pitch,
            pitch=-yaw,
            yaw=-roll,
            order=[1, 0, 2],
            z_down=z_down, 
        ).permute(1, 0)
        n2 = torch.cross(os_v_r, oa_v_r)
        n2 = torch.nn.functional.normalize(n2, p=2, dim=-1)
        cos_gamma = (n1 * n2).sum(dim=-1)
        sin_gamma = (torch.cross(n1, n2) * os_v_r).sum(dim=-1)
        self.tensor[..., -1] = torch.atan2(sin_gamma, cos_gamma)
        self.tensor[..., -1] = normalize(self.tensor[..., -1], min_=-torch.pi / 2, max_=torch.pi / 2)
        
        self.regularize_boxes(pattern='le90')


@register_box('shbox')
class SphereHBoxes(SphereBoxes):
    def __init__(
        self, 
        data,
        *args,
        with_norm: bool = True,
        with_regular: bool = False,
        clone: bool = True,
        **kwargs
    ) -> None:
        """ Sbox with angle always zero. """
        # Check if angle is present; if not, set it to 0.
        data = torch.as_tensor(data)
        if data.shape[-1] == 4 and data.numel():
            data = torch.cat([data, torch.zeros_like(data[..., [0]])], dim=-1)
            
        super().__init__(
            data, 
            *args, 
            with_norm=with_norm,
            with_regular=with_regular,
            clone=clone,
            **kwargs
        )
 