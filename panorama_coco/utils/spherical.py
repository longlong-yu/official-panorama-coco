"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-01-26
@desc:   
"""
import math
import cv2
from typing import List, Tuple, Union
from numpy import ndarray
from torch import Tensor
import torch
import torch.nn.functional as F

from panorama_coco.const.transforms import ExtendType

FLOAT_EPS = torch.finfo(torch.float).eps


class ImageUtils(object):
    """ Planar image utils. """
    @staticmethod
    def _merge_masks(masks: List[torch.Tensor]) -> Tensor:
        """ Merge masks into a single set of points. """
        objs = []
        for mask in masks:
            objs.append(mask.reshape(-1, 2))
        return torch.cat(objs)
    
    @classmethod
    def rbox_from_masks(cls, masks: List[torch.Tensor]) -> Tensor:
        """ Compute rotated bounding box (rbox) from the mask. """
        objs = cls._merge_masks(masks)
        (x, y), (w, h), angle = cv2.minAreaRect(objs.numpy())
        return torch.tensor([x, y, w, h, angle / 180 * torch.pi], device=masks[0].device)

    @classmethod
    def convex_hull(
        cls, 
        masks: List[torch.Tensor],
        clockwise: bool = False,
    ) -> Tensor:
        """ Return the convex hull point set. """
        objs = cls._merge_masks(masks)
        points = cv2.convexHull(
            points=objs.numpy(),
            clockwise=clockwise,
            returnPoints=True
        )
        return torch.tensor(points, device=objs.device).reshape(-1, 2)

    @staticmethod
    def rbox2corner(boxes: Tensor) -> Tensor:
        """
        refer: RotatedBoxes.rbox2corner
        Convert rotated box (x, y, w, h, t) to corners ((x1, y1), (x2, y1),
        (x1, y2), (x2, y2)).

        Args:
            boxes (Tensor): Rotated box tensor with shape of (..., 5).

        Returns:
            Tensor: Corner tensor with shape of (..., 4, 2).
        """
        ctr, w, h, theta = torch.split(boxes, (2, 1, 1, 1), dim=-1)
        cos_value, sin_value = torch.cos(theta), torch.sin(theta)
        vec1 = torch.cat([w / 2 * cos_value, w / 2 * sin_value], dim=-1)
        vec2 = torch.cat([-h / 2 * sin_value, h / 2 * cos_value], dim=-1)
        pt1 = ctr + vec1 + vec2
        pt2 = ctr + vec1 - vec2
        pt3 = ctr - vec1 - vec2
        pt4 = ctr - vec1 + vec2
        return torch.stack([pt1, pt2, pt3, pt4], dim=-2)
    
    @staticmethod
    def pxpy2xy(
        pxpy: Union[Tensor, Tuple[Tensor, Tensor]],
        w: Union[int, Tensor], 
        h: Union[int, Tensor]
    ) -> Tensor:
        """ Convert pixel coordinates to coordinate system coordinates. """
        x = (pxpy[0] / (w - 1) - 0.5) * w
        y = (0.5 - pxpy[1] / (h - 1)) * h
        return torch.stack((x, y))
    
    @staticmethod
    def xy2pxpy(
        xy: Union[Tensor, Tuple[Tensor, Tensor]],
        w: Union[int, Tensor], 
        h: Union[int, Tensor]
    ) -> Tensor:
        """ Convert coordinate system coordinates to pixel coordinates. """
        px = (xy[0] / w + 0.5) * (w - 1)
        px = torch.clamp(px.round(), 0, w - 1)
        py = (0.5 - xy[1] / h) * (h - 1)
        py = torch.clamp(py.round(), 0, h - 1)
        return torch.stack((px, py)).int()
    
    @classmethod
    def center_square_pad(
        cls,
        input: ndarray, 
        extend_type: ExtendType, 
        size_divisor: int = 1
    ) -> Tuple[Tensor, Tuple]:
        """ 
        pad the input to square and keep the center of input unchanged.
        
        input: (h, w, c)
        """
        h, w = input.shape[:2]
        aligned_h = math.ceil(h / size_divisor) * size_divisor
        aligend_w = math.ceil(w / size_divisor) * size_divisor
        edge = max(aligned_h, aligend_w)
        pad_h = edge - h
        pad_w = edge - w
        pad = (
            math.floor(pad_w / 2),
            math.ceil(pad_w / 2),
            math.floor(pad_h / 2),
            math.ceil(pad_h / 2),
        )
        return cls.bias_pad(input=input, pad=pad, extend_type=extend_type), pad
     
    @staticmethod
    def bias_pad(
        input: ndarray,
        pad: Tuple[int],
        extend_type: ExtendType,
    ) -> Tensor:
        """
        pad: (left, right, up, down)
        """
        if extend_type == ExtendType.ZERO:
            border_type = cv2.BORDER_CONSTANT
        elif extend_type == ExtendType.REFLECT:
            border_type = cv2.BORDER_REFLECT_101
        elif extend_type == ExtendType.CIRCULAR:
            border_type = cv2.BORDER_WRAP
        elif extend_type == ExtendType.REPLICATE:
            border_type = cv2.BORDER_REPLICATE
        
        return cv2.copyMakeBorder(
            input,
            pad[2],
            pad[3],
            pad[0],
            pad[1],
            border_type,
            value=0,
        )
    
    @staticmethod
    def xyxy2xywh(boxes):
        x1, y1, x2, y2 = torch.chunk(boxes, 4, dim=1) # Nx1
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = x2 - x1
        h = (y2 - y1)
        return torch.cat([x, y, w, h], dim=1) # Nx4

    @staticmethod
    def xywh2xyxy(boxes):
        x, y, w, h = torch.chunk(boxes, 4, dim=1) # Nx1
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return torch.cat([x1, y1, x2, y2], dim=1) # Nx4

    
class SphereImageUtils(object):
    """ Sphere image utils. """
    @staticmethod
    def xy2theta_phi_gnomonic(
        xy: Union[Tensor, Tuple[Tensor, Tensor]],
        radius: Union[int, Tensor], 
    ) -> Tensor:
        """ gnomonic project """
        theta = torch.atan(xy[0] / radius)
        phi = torch.atan2(xy[1], torch.sqrt(xy[0] ** 2 + radius ** 2))
        return torch.stack((theta, phi))

    @staticmethod
    def _calculate_sbox(
        points: Tensor,
        up_normal: Tensor,
        down_normal: Tensor,
    ) -> Tensor:
        """ Determine center point, field of view angles, rotation angle, and area. """
        up_vector = torch.tensor((0., 1, 0), device=points.device)
         
        v_normal = torch.cross(down_normal, up_normal)
        v_normal = F.normalize(v_normal, p=2.0, dim=0) 
        
        up_mid = torch.cross(up_normal, v_normal) 
        up_mid = F.normalize(up_mid, p=2.0, dim=0)     
        down_mid = torch.cross(v_normal, down_normal)
        down_mid = F.normalize(down_mid, p=2.0, dim=0)
        ct = up_mid + down_mid
        ct = F.normalize(ct, p=2.0, dim=0)
        beta = torch.acos(up_mid @ down_mid)
        
        # Positive direction points upward.
        h_normal = torch.cross(ct, v_normal)
        h_normal = F.normalize(h_normal, p=2.0, dim=0)
        min_cos = None
        for j in range(points.shape[0]):
            j_normal = torch.cross(h_normal, points[j])
            j_normal = F.normalize(j_normal, p=2.0, dim=0)
            cos = j_normal @ v_normal
            if min_cos is None or cos < min_cos:
                min_cos = cos
        alpha = 2 * torch.acos(min_cos)
        
        h_vector = torch.cross(up_vector, ct)
        h_vector = F.normalize(h_vector, p=2.0, dim=0)
        cos = h_vector @ v_normal
        sin = torch.cross(h_vector, v_normal) @ ct
        if cos < 0:
            gamma = torch.atan2(-sin, -cos)
        else:
            gamma = torch.atan2(sin, cos)
        
        return [ct, alpha, beta, gamma]
     
    @classmethod
    def sbox_from_convex_hull(
        cls,
        points: Tensor,
    ) -> Tensor:
        """ 
        Compute the minimum-area bounding rectangle from the convex hull.

        Args:
        points: (N, 3). Convex hull vertices are ordered counterclockwise and represented as unit vectors.
        """
        eps = FLOAT_EPS
        n = points.shape[0]
        next_n = lambda e: (e + 1) % n
        before_n = lambda e: (e - 1) % n
        
        sbox = None
        area = None
        for i in range(n):
            point = points[i]
            next_i = next_n(i)
            next_point = points[next_i]
            
            normal = torch.cross(point, next_point)
            normal = F.normalize(normal, p=2.0, dim=0)
        
            is_right = True
            
            # Traverse adjacent edges in reverse order.
            for k in range(n - 1):
                k = (next_i + k) % n
                next_k = next_n(k)
                
                point_k = points[k]
                next_point_k = points[next_k]
                
                normal_k = torch.cross(point_k, next_point_k)
                normal_k = F.normalize(normal_k, p=2.0, dim=0)
                
                point_k_2 = torch.cross(normal_k, normal)
                point_k_2 = F.normalize(point_k_2, p=2.0, dim=0) 
                point_k_2 = point_k_2 * (point_k_2 @ point_k)
                # 1 edge + 3 vertices
                if next_k != i:
                    for p in range(n):
                        p = next_n(k + p)
                        if p == i:
                            break
                        point_D = points[p]
                        normal_D = torch.cross(normal, point_D)
                        point_D_2 = torch.cross(normal_D, normal)
                        point_D_2 = F.normalize(point_D_2, p=2.0, dim=0) 
                        point_D_2 = point_D_2 * (point_D_2 @ point_D)
                        for q in range(n):
                            q = next_n(p + q)
                            if before_n(q) == i:
                                break
                            
                            if (k == next_i and q != i) or (k != next_i and q == i):
                                continue
                            
                            point_F = points[q]
                            if k == next_i and q == i:
                                point_P = point_k + point_F
                            else:
                                normal_F = torch.cross(normal, point_F)
                                point_F_2 = torch.cross(normal_F, normal)
                                point_F_2 = F.normalize(point_F_2, p=2.0, dim=0) 
                                point_F_2 = point_F_2 * (point_F_2 @ point_F)
                                
                                angle_add = torch.acos(point_k_2 @ point_F_2)
                                tmp1 = 1 / point_k_2.norm()
                                tmp2 = (point_F @ normal) / (point_k @ normal) / point_F_2.norm()
                                angle_minus = torch.atan(
                                    torch.tan(angle_add / 2) * (tmp1 - tmp2) / (tmp1 + tmp2)
                                )
                                angle1 = (angle_add + angle_minus) / 2
                                if torch.cross(point_k_2, point_F_2) @ normal >= 0:
                                    point_P = cls.rotate(
                                        n=normal,
                                        xyz=point_k_2.unsqueeze(dim=-1),
                                        gamma=angle1
                                    ).squeeze()
                                else:
                                    point_P = cls.rotate(
                                        n=normal,
                                        xyz=point_k_2.unsqueeze(dim=-1),
                                        gamma=-angle1
                                    ).squeeze()
                                if point_P @ point < 0:
                                    point_P = -point_P 
                            v_normal = torch.cross(normal, point_P)
                            up_normal = torch.cross(v_normal, point_D)
                            up_normal = F.normalize(up_normal, p=2.0, dim=0)
                            
                            # Check whether the PE covers adjacent points.
                            next_point_D = points[next_n(p)] 
                            if next_point_D @ up_normal < 0:
                                continue
                            before_point_D = points[before_n(p)]
                            if before_point_D @ up_normal < 0:
                                continue
                            
                            new_sbox = cls._calculate_sbox(
                                points=points,
                                up_normal=up_normal,
                                down_normal=normal,
                            ) 
                            new_area = 4 * torch.asin(torch.sin(new_sbox[1] / 2) * torch.sin(new_sbox[2] / 2))
                            if area is None or new_area < area:
                                area = new_area
                                sbox = new_sbox
                
                # 2 edge + 1 vertices
                eit = torch.pi - torch.acos(normal_k @ normal)
                # Case with fixed right adjacent edge.
                if is_right and eit > torch.pi / 2:
                    new_area = 4 * eit - 2 * torch.pi
                    if area is not None and new_area >= area:
                        continue
                    
                    # Fix the vertex E on the edge opposite to AB.
                    for p in range(n):
                        p = (next_k + p) % n
                        # Stop traversal when reaching edge AB.
                        if p == i:
                            break
                        
                        point_E = points[p]
                        angle_E = torch.pi / 2 - torch.acos(point_E @ normal_k)
                        if angle_E > torch.pi - eit:
                            continue
                        
                        # If E and D coincide.
                        if p == next_k:
                            point_P = cls.rotate(
                                n=point_E,
                                xyz=normal_k.unsqueeze(dim=-1),
                                gamma=torch.pi / 2 - eit,
                            ).squeeze()
                            point_P = F.normalize(point_P, p=2.0, dim=0)
                            up_normal = torch.cross(point_E, point_P)
                            up_normal = F.normalize(up_normal, p=2.0, dim=0)
                        else:
                            normal_E = torch.cross(normal_k, point_E)
                            point_E_2 = torch.cross(normal_E, normal_k)
                            point_E_2 = F.normalize(point_E_2, p=2.0, dim=0) 
                            point_E_2 = point_E_2 * (point_E_2 @ point_E)
                            angle_E = torch.asin(point_E @ normal_k / torch.tan(-eit) / point_E_2.norm())
                            point_P = cls.rotate(
                                n=normal_k,
                                xyz=point_E_2.unsqueeze(dim=-1),
                                gamma=-angle_E
                            ).squeeze()
                            point_P = F.normalize(point_P, p=2.0, dim=0)
                            up_normal = torch.cross(point_P, point_E)
                            up_normal = F.normalize(up_normal, p=2.0, dim=0)
                        
                        # Check whether the PE covers adjacent points.
                        next_point_E = points[next_n(p)] 
                        if next_point_E @ up_normal < 0:
                            continue
                        before_point_E = points[before_n(p)]
                        if before_point_E @ up_normal < 0:
                            continue
                        
                        new_sbox = cls._calculate_sbox(
                            points=points,
                            up_normal=up_normal,
                            down_normal=normal,
                        ) 
                        new_area = 4 * torch.asin(torch.sin(new_sbox[1] / 2) * torch.sin(new_sbox[2] / 2))
                        if area is None or new_area < area:
                            area = new_area
                            sbox = new_sbox
                      
                    # Fix vertex F on the edge opposite to CD.
                    for q in range(n):
                        q = next_n(next_k + q)
                        # Stop traversal when reaching edge AB.
                        if q == (i + 1) % n:
                            break
                        
                        point_F = points[q]
                        angle_F = torch.pi / 2 - torch.acos(point_F @ normal)
                        if angle_F > torch.pi - eit:
                            continue
                        
                         # If F and A coincide.
                        if q == i or point_F @ normal < eps:
                            point_P = cls.rotate(
                                n=point_F,
                                xyz=normal_k.unsqueeze(dim=-1),
                                gamma=eit - torch.pi / 2,
                            ).squeeze()
                            point_P = F.normalize(point_P, p=2.0, dim=0)
                            up_normal = torch.cross(point_P, point_F)
                            up_normal = F.normalize(up_normal, p=2.0, dim=0)
                        else:   
                            normal_F = torch.cross(normal, point_F)
                            point_F_2 = torch.cross(normal_F, normal)
                            point_F_2 = F.normalize(point_F_2, p=2.0, dim=0)
                            point_F_2 = point_F_2 * (point_F_2 @ point_F)
                            angle_F = torch.asin(point_F @ normal / torch.tan(-eit) / point_F_2.norm())
                            point_P = cls.rotate(
                                n=normal,
                                xyz=point_F_2.unsqueeze(dim=-1),
                                gamma=angle_F
                            ).squeeze()
                            point_P = F.normalize(point_P, p=2.0, dim=0)
                            up_normal = torch.cross(point_F, point_P)
                            up_normal = F.normalize(up_normal, p=2.0, dim=0)
                        # Check whether FP covers adjacent points.
                        next_point_F = points[next_n(q)] 
                        if next_point_F @ up_normal < 0:
                            continue
                        before_point_F = points[before_n(q)]
                        if before_point_F @ up_normal < 0:
                            continue
                        
                        new_sbox = cls._calculate_sbox(
                            points=points,
                            up_normal=up_normal,
                            down_normal=normal_k,
                        ) 
                        new_area = 4 * torch.asin(torch.sin(new_sbox[1] / 2) * torch.sin(new_sbox[2] / 2))
                        if area is None or new_area < area:
                            area = new_area
                            sbox = new_sbox
                # Case with fixed opposite edge.
                else:
                    if eit > torch.pi / 2:
                        break
                    is_right = False
                    new_sbox = cls._calculate_sbox(
                        points=points,
                        up_normal=normal_k,
                        down_normal=normal,
                    )
                    new_area = 4 * torch.asin(torch.sin(new_sbox[1] / 2) * torch.sin(new_sbox[2] / 2))
                    if area is None or new_area < area:
                        area = new_area
                        sbox = new_sbox
            
        theta_phi = cls.xyz2theta_phi(sbox[0])
        return torch.tensor(theta_phi.tolist() + sbox[1:], device=points.device)
   
    @classmethod
    def sbox_area(cls, sbox: Tensor, radius: float = 1) -> Tensor:
        """ Compute the area of the sbox. """
        return cls.fov_area(
            alpha=sbox[..., 2], 
            beta=sbox[..., 3], 
            radius=radius
        )
    
    @staticmethod
    def fov_area(alpha: Tensor, beta: Tensor, radius: float = 1) -> Tensor:
        """ ompute area based on field of view angles. """
        return 4 * radius * radius * torch.asin(torch.sin(alpha / 2) * torch.sin(beta / 2))
    
    @staticmethod
    def theta_phi2pxpy(
        theta_phi: Union[Tensor, Tuple[Tensor, Tensor]], 
        w: Union[int, Tensor], 
        h: Union[int, Tensor],
        clamp: bool = True,
    ) -> Tensor:
        """ 
        Args:
            clamp: whether to ensure the return data to be int
        """
        px = (theta_phi[0] / torch.pi + 1) * 0.5 * (w - 1)
        py = (0.5 - theta_phi[1] / torch.pi) * (h - 1)
        if clamp:
            px = torch.clamp(px.round(), 0, w - 1)
            py = torch.clamp(py.round(), 0, h - 1)
            return torch.stack((px, py)).int()
        else:
            return torch.stack((px, py)) 
    
    @staticmethod
    def pxpy2theta_phi(
        pxpy: Union[Tensor, Tuple[Tensor, Tensor]],
        w: Union[int, Tensor], 
        h: Union[int, Tensor]
    ) -> Tensor:
        theta = (pxpy[0] / (w - 1) - 0.5) * 2 * torch.pi
        phi = (0.5 - pxpy[1] / (h - 1)) * torch.pi
        return torch.stack((theta, phi))
    
    @staticmethod
    def theta_phi2xyz(theta_phi: Union[Tensor, Tuple[Tensor, Tensor]]) -> Tensor:
        x_3d = torch.cos(theta_phi[1]) * torch.sin(theta_phi[0])
        y_3d = torch.sin(theta_phi[1])
        z_3d = torch.cos(theta_phi[1]) * torch.cos(theta_phi[0])
        return torch.stack((x_3d, y_3d, z_3d))

    @staticmethod
    def xyz2theta_phi(
        xyz: Union[Tensor, Tuple[Tensor, Tensor, Tensor]]
    ) -> Tensor:
        theta = torch.atan2(xyz[0], xyz[2])
        phi = torch.atan2(xyz[1], torch.sqrt(xyz[0] ** 2 + xyz[2] ** 2))
        return torch.stack((theta, phi))
    
    @classmethod
    def pxpy2xyz(
        cls,
        pxpy: Union[Tensor, Tuple[Tensor, Tensor]],
        w: Union[int, Tensor], 
        h: Union[int, Tensor]
    ) -> Tensor:
        return cls.theta_phi2xyz(
            cls.pxpy2theta_phi(pxpy, w, h)
        )
    
    @classmethod  
    def xyz2pxpy(
        cls,
        xyz: Union[Tensor, Tuple[Tensor, Tensor, Tensor]],
        w: Union[int, Tensor], 
        h: Union[int, Tensor] 
    ) -> Tensor:
        return cls.theta_phi2pxpy(
            cls.xyz2theta_phi(xyz), w, h
        )
    
    @staticmethod
    def rotate(
        n: Tensor, 
        xyz: Tensor, 
        gamma: Union[Tensor, float] = 0
    ) -> Tensor:
        """ 
        n must be a unit vector.
        For gamma, counterclockwise is positive. 
        """
        n11 = (n[0] ** 2) * (1 - torch.cos(gamma)) + torch.cos(gamma)
        n12 = n[0] * n[1] * (1 - torch.cos(gamma)) - n[2] * torch.sin(gamma)
        n13 = n[0] * n[2] * (1 - torch.cos(gamma)) + n[1] * torch.sin(gamma)

        n21 = n[0] * n[1] * (1 - torch.cos(gamma)) + n[2] * torch.sin(gamma)
        n22 = (n[1] ** 2) * (1 - torch.cos(gamma)) + torch.cos(gamma)
        n23 = n[1] * n[2] * (1 - torch.cos(gamma)) - n[0] * torch.sin(gamma)

        n31 = n[0] * n[2] * (1 - torch.cos(gamma)) - n[1] * torch.sin(gamma)
        n32 = n[1] * n[2] * (1 - torch.cos(gamma)) + n[0] * torch.sin(gamma)
        n33 = (n[2] ** 2) * (1 - torch.cos(gamma)) + torch.cos(gamma)

        x, y, z = xyz[0], xyz[1], xyz[2]
        xx = n11 * x + n12 * y + n13 * z
        yy = n21 * x + n22 * y + n23 * z
        zz = n31 * x + n32 * y + n33 * z

        return torch.stack((xx, yy, zz))

    @staticmethod
    def rotate_xyz(
        xyz: Tensor, 
        *, 
        roll: Union[float, Tensor], 
        pitch: Union[float, Tensor], 
        yaw: Union[float, Tensor], 
        z_down: bool = True,
        order: List[int] = [0, 1, 2],
    ) -> Tensor:
        """Create Rotation Matrix

        params:
        - roll, pitch, yaw (float): in radians
        - z_down (bool): flips pitch and yaw directions

        returns:
        - R (torch.Tensor): 3x3 rotation matrix
        """
        if not torch.is_tensor(roll):
            roll = xyz.new_tensor(roll)
            pitch = xyz.new_tensor(pitch)
            yaw = xyz.new_tensor(yaw)
        
        # calculate rotation about the x-axis
        R_x = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, torch.cos(roll), -torch.sin(roll)],
            [0.0, torch.sin(roll), torch.cos(roll)],
        ], dtype=xyz.dtype, device=xyz.device)
        # calculate rotation about the y-axis
        if not z_down:
            pitch = -pitch
        R_y = torch.tensor([
            [torch.cos(pitch), 0.0, torch.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-torch.sin(pitch), 0.0, torch.cos(pitch)],
        ], dtype=xyz.dtype, device=xyz.device)
        # calculate rotation about the z-axis
        if not z_down:
            yaw = -yaw
        R_z = torch.tensor([
            [torch.cos(yaw), -torch.sin(yaw), 0.0],
            [torch.sin(yaw), torch.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=xyz.dtype, device=xyz.device)
        Rs = [R_x, R_y, R_z]
        Rs = [Rs[item] for item in order]
        return Rs[2] @ Rs[1] @ Rs[0] @ xyz
    
    @staticmethod
    def rad2angle(rad: Union[float, int, Tensor]) -> Union[float, Tensor]:
        return rad * 180 / torch.pi

    @staticmethod
    def angle2rad(angle: Union[float, int, Tensor]) -> Union[float, Tensor]:
        return angle * torch.pi / 180

    @staticmethod
    def gaussian_radius(sbox: Tensor, threshold: float = 0.7) -> float:
        """ Compute the Gaussian radius based on the formula from arXiv:2108.08029v1. """
        tmp1 = torch.acos(-torch.sin(sbox[2]/2) * torch.sin(sbox[3]/2))
        tmp2 = tmp1 - torch.pi / 2
        
        tmp = -2 * torch.sin(tmp2 / threshold) + torch.cos((sbox[2] - sbox[3])/2)
        r1 = 0.5 * torch.acos(tmp) - (sbox[2] + sbox[3]) / 4
        
        tmp = -2 * torch.sin(threshold * tmp2) + torch.cos((sbox[2] - sbox[3])/2)
        r2 = -0.5 * torch.acos(tmp) + (sbox[2] + sbox[3]) / 4
        
        tmp = -2 * torch.sin(2 * threshold / (1 + threshold) * (tmp1 - 2 * torch.pi)) + torch.cos((sbox[2] - sbox[3])/2)
        r3 = -torch.acos(tmp) + (sbox[2] + sbox[3]) / 2 

        return min([item.item() for item in (r1, r2, r3) if not item.isnan() and item > 0])
        
    @classmethod
    def gen_gaussian_target(
        cls,
        center_heatmap: Tensor, 
        sboxes: Tensor,
        threshold: float = 0.3,
        auto_ajust: bool = False,
    ) -> Tensor:
        """ Generate center-point Gaussian distributions based on sboxes; algorithm referenced from PANDORA. """
        h, w = center_heatmap.shape[-2:]
        normals = cls.sboxes_normals(sboxes)
        
        xs = torch.linspace(
            -torch.pi, 
            torch.pi, 
            steps=w, 
            device=center_heatmap.device, 
            dtype=center_heatmap.dtype
        )
        ys = torch.linspace(
            torch.pi / 2, 
            -torch.pi / 2, 
            steps=h, 
            device=center_heatmap.device, 
            dtype=center_heatmap.dtype
        )
        ys, xs = torch.meshgrid([ys, xs], indexing='ij')
        theta_phis = torch.stack((xs, ys), dim=-1).permute(2, 0, 1)
        xyzs = cls.theta_phi2xyz(theta_phis.view(2, -1)).view(3, h, w).permute(1, 2, 0)
        sboxes = sboxes.permute(1, 0)
        cts = cls.theta_phi2xyz(sboxes[0:2]).permute(1, 0)
        
        # Dynamically adjust the radius.
        if auto_ajust:
            areas = SphereImageUtils.sbox_area(sboxes.permute(1, 0))
            ratios = -torch.log10(areas).int()
            ratios = torch.clamp(ratios, min=0, max=5)
            threshold = cts.new_tensor([threshold] * sboxes.shape[-1]) / (ratios * 2 + 1)
        
        sigma_ws = torch.tan(sboxes[2] / 2)
        sigma_ws = cls.gaussian2D_radius(det_size=(sigma_ws, sigma_ws), min_overlap=threshold)
        sigma_hs = torch.tan(sboxes[3] / 2)
        sigma_hs = cls.gaussian2D_radius(det_size=(sigma_hs, sigma_hs), min_overlap=threshold)
        
        for ct, normal, sigma_w, sigma_h in zip(
            cts, normals, sigma_ws, sigma_hs
        ):
            u_mask = xyzs @ normal[0] >= 0
            r_mask = xyzs @ normal[1] >= 0
            d_mask = xyzs @ normal[2] >= 0
            l_mask = xyzs @ normal[3] >= 0
            mask = u_mask & r_mask & d_mask & l_mask
            
            target_heatmap = torch.zeros_like(center_heatmap)
            
            masked_xyzs = xyzs[mask] / (xyzs[mask] @ ct)[..., None].expand(-1, 3)
            masked_xyzs -= ct
            
            masked_w = masked_xyzs @ normal[4] 
            maksed_h = masked_xyzs @ normal[5]
             
            target_heatmap[mask] = (-0.5 * ((masked_w / sigma_w) ** 2 + (maksed_h / sigma_h) ** 2)).exp() 
            
            # The point with the maximum probability must be set to 1; otherwise, the positive loss in Gaussian focal loss will always be zero.
            tmp = target_heatmap.flatten()
            tmp[torch.argmax(tmp)] = 1.
            target_heatmap = tmp.view_as(target_heatmap)
            
            target_heatmap[target_heatmap < torch.finfo(target_heatmap.dtype).eps * target_heatmap.max()] = 0
        
            torch.max(center_heatmap, target_heatmap, out=center_heatmap)
        return center_heatmap
   
    @classmethod
    def sboxes_normals(cls, sboxes: Tensor) -> Tensor:
        """ 
        Compute the normal vectors from the sbox and return them in the order: Up, Right, Down, Left. 
        Counterclockwise is positive, and normals point inward to the sbox. 
        """
        sboxes = sboxes.permute(1, 0)
        cts = cls.theta_phi2xyz(sboxes[0:2]).permute(1, 0)
        up_vector = cts.new_tensor([0., 1, 0])
        
        h_normals = torch.cross(up_vector[None], cts, dim=-1)
        h_normals = F.normalize(h_normals, p=2, dim=-1)
        h_normals = cls.rotate(
            n=cts.permute(1, 0),
            xyz=h_normals.permute(1, 0),
            gamma=sboxes[4],
        )
      
        v_normals = torch.cross(cts, h_normals.permute(1, 0), dim=-1)
        v_normals = F.normalize(v_normals, p=2, dim=-1).permute(1, 0)
        
        down_normals = cls.rotate(
            n=h_normals,
            xyz=v_normals,
            gamma=sboxes[3] / 2
        )
        up_normals = cls.rotate(
            n=h_normals,
            xyz=-v_normals,
            gamma=-sboxes[3] / 2
        )
        l_normals = cls.rotate(
            n=v_normals,
            xyz=h_normals,
            gamma=-sboxes[2] / 2,
        )
        r_normals = cls.rotate(
            n=v_normals,
            xyz=-h_normals,
            gamma=sboxes[2] / 2,
        )
        normals = torch.stack((
            up_normals, r_normals, 
            down_normals, l_normals, 
            h_normals, v_normals
        ), dim=1)
        return F.normalize(normals.permute(2, 1, 0), p=2, dim=-1)

    @staticmethod
    def gaussian2D_radius(
        det_size: Union[Tensor, Tuple[Tensor, Tensor]], 
        min_overlap: Union[float, Tensor] = 0.3
    ) -> Tensor:
        """
        Generate 2D gaussian radius.
        Modified from mmdet.utils.gaussian_radius to support Tensor.
        
        Args:
            det_size: (H, W)，shape is (N, 2).
        """
        height, width = det_size

        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = torch.sqrt(b1**2 - 4 * a1 * c1)
        r1 = (b1 - sq1) / (2 * a1)

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = torch.sqrt(b2**2 - 4 * a2 * c2)
        r2 = (b2 - sq2) / (2 * a2)

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = torch.sqrt(b3**2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / (2 * a3)
        return torch.min(torch.stack((r1, r2, r3), dim=-1), dim=-1)[0]

    @classmethod
    def fov2sbox(cls, fov: Union[Tensor, list]) -> Tensor:
        """
        chenbin-fasion to sbox.
        chenbin-fasion: ([0, 360), [0, 180), [0, 360], [0, 180]), radian
        
        fov: [..., 4/5]
        """
        if torch.is_tensor(fov):
            fov[..., 0] = cls.angle2rad(fov[..., 0] - 180)
            fov[..., 1] = cls.angle2rad(90 - fov[..., 1])
            fov[..., 2] = cls.angle2rad(fov[..., 2])
            fov[..., 3] = cls.angle2rad(fov[..., 3])
        else:
            fov[0] = cls.angle2rad(fov[0] - 180)
            fov[1] = cls.angle2rad(90 - fov[1])
            fov[2] = cls.angle2rad(fov[2])
            fov[3] = cls.angle2rad(fov[3]) 
        return fov

    @classmethod
    def sbox2fov(cls, fov: Union[Tensor, list]) -> Tensor:
        """
        sbox to chenbin-fasion.
        chenbin-fasion: ([0, 360), [0, 180), [0, 360], [0, 180]), radian
        
        fov: [..., 4/5]
        """
        if torch.is_tensor(fov):
            fov[..., 0] = cls.rad2angle(fov[..., 0]) +  180
            fov[..., 1] = 90 - cls.rad2angle(fov[..., 1])
            fov[..., 2] = cls.rad2angle(fov[..., 2])
            fov[..., 3] = cls.rad2angle(fov[..., 3])
        else:
            fov[0] = cls.rad2angle(fov[0]) + 180
            fov[1] = 90 - cls.rad2angle(fov[1])
            fov[2] = cls.rad2angle(fov[2])
            fov[3] = cls.rad2angle(fov[3]) 
        return fov
    
    @staticmethod
    def erp_weights(device, erp_h: int, erp_w: int) -> Tensor:
        """ 
        Weights derived from spherical differential area.
        refer: UNBIASED IOU FOR SPHERICAL IMAGE OBJECT DETECTION
        """
        y = torch.linspace(0.5, -0.5, erp_h, device=device)
        x = torch.linspace(-1, 1, erp_w, device=device)
        y, _ = torch.meshgrid([y, x], indexing='ij')
        return (torch.cos(y * torch.pi / erp_h) - torch.cos((y + 1) * torch.pi / erp_h)) * 2 * torch.pi / erp_w

    @staticmethod
    def sbox_h_alpha_beta(sboxes: Tensor) -> Tensor:
        # tangent plane
        w = 2 * torch.tan(sboxes[..., 2] / 2)
        h = 2 * torch.tan(sboxes[..., 3] / 2)
        h_w = w * torch.cos(sboxes[..., 4]).abs() + h * torch.sin(sboxes[..., 4]).abs()
        h_h = w * torch.sin(sboxes[..., 4]).abs() + h * torch.cos(sboxes[..., 4]).abs()
        h_alpha = 2 * torch.atan(h_w / 2)
        h_beta = 2 * torch.atan(h_h / 2)
        return h_alpha, h_beta
    
    @classmethod
    def sbox2erp(cls, sboxes: Tensor, estimate: bool = True):
        """ 
        Compute the horizontal bounding box on the ERP from the sbox.
        """
        sboxes[..., 2:4] = sboxes[..., 2:4].clamp_(
            min=0 + 2 *FLOAT_EPS,
            max=torch.pi - 2 * FLOAT_EPS,
        )
        h_alpha, h_beta = cls.sbox_h_alpha_beta(sboxes)
        
        # Determine the latitude with the longest longitudinal span.
        upper_phi = sboxes[..., 1] + h_beta / 2
        upper_phi = upper_phi.clamp_(
            min=-torch.pi / 2 + FLOAT_EPS,
            max=torch.pi / 2 - FLOAT_EPS
        )
        lower_phi = sboxes[..., 1] - h_beta / 2
        lower_phi = lower_phi.clamp_(
            min=-torch.pi / 2 + FLOAT_EPS,
            max=torch.pi / 2 - FLOAT_EPS
        )
        phi = torch.where(upper_phi.abs() > lower_phi.abs(), upper_phi, lower_phi)
        
        # Compute the longitude span at this latitude.
        if estimate:
            tmp = torch.sin(h_alpha / 2) / torch.cos(phi)
            h_alpha_2 = 2 * torch.asin(tmp.clamp_(min=0, max=1 - FLOAT_EPS))
        else:
            h_beta_half = (phi - sboxes[..., 1]).abs()
            tmp1 = torch.sin(h_alpha / 2)
            tmp2 = torch.cos(h_alpha / 2).clamp_min_(FLOAT_EPS)
            tmp3 = torch.sin(h_beta_half)
            tmp4 = torch.cos(h_beta_half)
            l = 2 * tmp1 * tmp4 * torch.sqrt(tmp3 ** 2 / tmp2 ** 2 + tmp4 ** 2)
            a = 4 * (torch.cos(phi) ** 4)
            c = - torch.sin(phi * 2) ** 2
            b = -c + l ** 2 - a
            tmp = (-b + torch.sqrt(b ** 2 - 4 * a * c)) / 2 / a.clamp_min_(FLOAT_EPS)
            h_alpha_2 = 2 * torch.acos(tmp.sqrt().clamp_(min=0, max=1 - FLOAT_EPS))
        
        sboxes[..., 2] = h_alpha_2
        sboxes[..., 3] = h_beta
        sboxes[..., 4] = 0
            
        return sboxes, h_alpha, h_beta
