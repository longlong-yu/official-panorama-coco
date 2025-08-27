"""
@author: longlong.yu, bin.wang
@email:  longlong.yu@hdu.edu.cn
@date:   2023-12-25
@desc:   
"""
from typing import Tuple, Union
import torch
import torch.nn.functional as F
from mmcv.transforms import BaseTransform
from panorama_coco.const.transforms import ExtendType
from panorama_coco.lib.equilib import cube2equi, equi2equi
from panorama_coco.registry import TRANSFORMS
from panorama_coco.utils.spherical import ImageUtils


@TRANSFORMS.register_module()
class SquareToERP(BaseTransform):
    """Convert planar image to ERP image.

    Required Keys:
    
    - img

    Modified Keys:
    
    - img
    - img_shape

    Added Keys:

    Args:
        width (int): Width of ERP image.
        height (int): Height of ERP image.
        mode (str): ('bilinear', 'bicubic', 'nearest')
        need_permute (bool): If input img is organzied as (H, W, C), this value should be set True.
        extend_mode (ExtendType): In which manner to extend cubemap from face.
    """
    
    def __init__(
        self, 
        width: int,
        height: int,
        mode: str='bilinear',
        need_permute: bool = True,
        extend_mode: ExtendType = ExtendType.REFLECT
    ) -> None:
        self.width = width
        self.height = height
        self.mode = mode
        self.need_permute = need_permute
        self.extend_mode = extend_mode

    def transform(self, results: dict, mode: str = None) -> dict:
        """
        The transform function.
        
        results:
            face_images: If results contain the key 'face_images', then the face images will 
                be directly used for other faces, and extend_mode will be ignored. The face 
                images should be all the same size.
        """
        img = self._prepare_img(results['img'])

        # draw bboxes
        # if results.get('gt_bboxes'):
        #     f_img = img.permute(1, 2, 0)
        #     f_img = mmcv.imshow_bboxes(
        #         img=f_img.cpu().numpy(),
        #         bboxes=results['gt_bboxes'].tensor.cpu().numpy(),
        #         show=False,
        #     )
        #     f_img = torch.from_numpy(f_img).to(device=device).permute(2, 0, 1)
        
        if results.get('face_images'):
            cubemap = {
                'F': img,
                'B': self._prepare_img(results['face_images'][0]),
                'L': self._prepare_img(results['face_images'][1]),
                'R': self._prepare_img(results['face_images'][2]),
                'U': self._prepare_img(results['face_images'][3]),
                'D': self._prepare_img(results['face_images'][4]),
            }
        elif self.extend_mode == ExtendType.ZERO:
            zero_img = torch.zeros_like(img)
            cubemap = {
                'F': img,
                'B': zero_img,
                'L': zero_img,
                'R': zero_img,
                'U': zero_img,
                'D': zero_img,
            }
        elif self.extend_mode == ExtendType.REFLECT:
            flip_img = img.flip(dims=(2,))
            cubemap = {
                'F': img,
                'B': img,
                'L': flip_img,
                'R': flip_img,
                'U': self._build_up_reflect(img),
                'D': self._build_down_reflect(img),
            }
        elif self.extend_mode == ExtendType.CIRCULAR:
            cubemap = {
                'F': img,
                'B': img,
                'L': img,
                'R': img,
                'U': self._build_up_circular(img),
                'D': self._build_down_circular(img),
            }
        else:
            raise Exception(f'Invalid ExtendType {self.extend_type} !')
            
        img = cube2equi(
            cubemap=cubemap,
            cube_format='dict',
            width=self.width,
            height=self.height,
            mode=mode if mode else self.mode,
            backend='native',
        )
        results['img_shape'] = img.shape[-2:]
        results['img'] = self._recover_img(img, results['img'])
                   
        return results

    def _prepare_img(self, image):
        """ Prepare the image so that it matches the requirements of equilib. """ 
        # device = torch_device()
        device = torch.device('cpu')
        if torch.is_tensor(image):
            image = image
        else:
            image = torch.from_numpy(image).to(device=device)
        
        # (H, W, C) -> (C, H, W)
        if self.need_permute:
            image = image.permute(2, 0, 1)
        return image

    def _recover_img(self, image, origin_image):
        """ Recover the image. """
        if self.need_permute:
            image = image.permute(1, 2, 0)
        if not torch.is_tensor(origin_image):
            image = image.cpu().numpy()
        return image
    
    @staticmethod
    def _build_up_reflect(img: torch.Tensor) -> torch.Tensor:
        """Build up face of the cube."""
        half_w = img.size()[1] // 2
        mask = img.new_ones((half_w, half_w))
        mask = mask.triu(diagonal=1) + torch.diag(img.new_ones((half_w,))) / 2
        mask = torch.cat([mask, mask.flip(dims=(1,))], dim=1)
        mask = F.pad(mask, (0, 0, 0, half_w))
        ret = (img * mask).flip(dims=(1,))
        ret += ret.transpose(1, 2).clone()
        return ret + ret.flip(dims=(1, 2))

    @classmethod
    def _build_down_reflect(cls, img: torch.Tensor) -> torch.Tensor:
        """Build down face of the cube."""
        return cls._build_up_reflect(img.flip(dims=(1, 2)))
    
    @staticmethod
    def _build_up_circular(img: torch.Tensor) -> torch.Tensor:
        """Build up face of the cube."""
        half_w = img.size()[1] // 2
        mask = img.new_ones((half_w, half_w))
        mask = mask.tril(diagonal=-1) + torch.diag(img.new_ones((half_w,))) / 2
        mask = torch.cat([mask.flip(dims=(1,)), mask], dim=1)
        mask = F.pad(mask, (0, 0, half_w, 0))
        ret = img * mask
        ret += ret.flip(dims=(2,)).transpose(1, 2).clone()
        return ret + ret.flip(dims=(1, 2))

    @classmethod
    def _build_down_circular(cls, img: torch.Tensor) -> torch.Tensor:
        """Build down face of the cube."""
        return cls._build_up_circular(img.flip(dims=(1, 2)))


@TRANSFORMS.register_module()
class RandomRotateERP(BaseTransform):
    """Rotate ERP image.

    Required Keys:
    
    - img
    - gt_bboxes (optional): (θ, φ, α, β, γ)

    Modified Keys:
    
    - img
    - gt_bboxes (optional): (θ, φ, α, β, γ)

    Added Keys:

    - sph_rot (dict): {'roll': float, 'pitch': float, 'yaw': float}, values represent radians.
    
    Args:
        mode (str): ('bilinear', 'bicubic', 'nearest') 
        need_permute (bool): If input img is organzied as (H, W, C), this value should be set True.
        angles (Tensor): Rotation axis.
    """
    
    def __init__(
        self, 
        mode: str='bilinear',
        need_permute: bool=True,
        angles: Union[torch.Tensor, float] = None,
        z_down: bool = False,
    ) -> None:
        self.mode = mode
        self.need_permute = need_permute
        self.angles = angles
        self.z_down = z_down
    
    def transform(self, results: dict) -> dict:
        """The transform function."""
        is_tensor = torch.is_tensor(results['img'])
        if is_tensor:
            img = results['img']
        else:
            img = torch.from_numpy(results['img'].copy()).cpu()
        if self.need_permute:
            img = img.permute(2, 0, 1)
        
        if self.angles is None:
            angles = torch.rand(3).cpu() * torch.pi * 2 
        else:
            angles = self.angles
        
        if not torch.is_tensor(angles):
            angles = torch.as_tensor(angles).cpu()
            
        results['sph_rot'] = {
            'roll': angles[0], 
            'pitch': angles[1], 
            'yaw': angles[2],
        }
        
        results['img'] = equi2equi(
            src=img, 
            rots=results['sph_rot'], 
            mode=self.mode, 
            z_down=self.z_down, 
            backend='native',
        )
        if self.need_permute:
            results['img'] = results['img'].permute(1, 2, 0)    
        if not is_tensor:
            results['img'] = results['img'].cpu().numpy()
        
        if 'gt_bboxes' in results:
            results['gt_bboxes'].rotate(angles[0], angles[1], angles[2], z_down=self.z_down)
         
        return results


@TRANSFORMS.register_module()
class Pad(BaseTransform):
    """ Pad image with boxes.

    Required Keys:
    
    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)

    Modified Keys:
    
    - img
    - img_shape
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)

    Added Keys:
    - pad: (left, right, up, down)

    Args:
        pad: (left, right, up, down)
    """
    
    def __init__(
        self, 
        pad: Tuple[int],
        extend_mode: ExtendType = ExtendType.REFLECT
    ) -> None:
        self.pad = pad
        self.extend_mode = extend_mode

    def transform(self, results: dict) -> dict:
        """
        The transform function.
        """
        results['img'] = ImageUtils.bias_pad(
            input=results['img'],
            pad=self.pad,
            extend_type=self.extend_mode,
        )
        results['img_shape'] = results['img'].shape[:2]
        results['pad'] = self.pad
        if 'gt_bboxes' in results:
            results['gt_bboxes'][..., 0] += self.pad[0]
            results['gt_bboxes'][..., 1] += self.pad[2] 
                   
        return results


@TRANSFORMS.register_module()
class CenterPad(BaseTransform):
    """ Pad image with boxes and keep image center unchanged.

    Required Keys:
    
    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)

    Modified Keys:
    
    - img
    - img_shape
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)

    Added Keys:
    - pad: (left, right, up, down)

    Args:
    """
    
    def __init__(
        self, 
        size_divisor: int = 1,
        extend_mode: ExtendType = ExtendType.REFLECT
    ) -> None:
        self.size_divisor = size_divisor
        self.extend_mode = extend_mode

    def transform(self, results: dict) -> dict:
        """
        The transform function.
        """
        results['img'], pad = ImageUtils.center_square_pad(
            results['img'],
            extend_type=self.extend_mode,
            size_divisor=self.size_divisor,
        )
        results['img_shape'] = results['img'].shape[:2]
        results['pad'] = pad
        if 'gt_bboxes' in results:
            results['gt_bboxes'].tensor[:, 0] += pad[0]
            results['gt_bboxes'].tensor[:, 1] += pad[2] 
                   
        return results


# todo @longlong.yu to validate
@TRANSFORMS.register_module()
class MaskedSquareToERP(BaseTransform):
    """Convert planar image to ERP image.

    Required Keys:
    
    - img

    Modified Keys:
    
    - img
    - img_shape

    Added Keys:

    Args:
        width (int): Width of ERP image.
        height (int): Height of ERP image.
        mode (str): ('bilinear', 'bicubic', 'nearest')
        need_permute (bool): If input img is organzied as (H, W, C), this value should be set True.
        extend_mode (ExtendType): In which manner to extend cubemap from face.
    """
    
    def __init__(
        self, 
        mode: str='bilinear',
        need_permute: bool = True,
    ) -> None:
        self.mode = mode
        self.need_permute = need_permute

    def transform(self, results: dict, mode: str = None) -> dict:
        """The transform function."""
        is_tensor = torch.is_tensor(results['img'])
        # device = torch_device()
        device = torch.device('cpu')
        if is_tensor:
            img = results['img']
        else:
            img = torch.from_numpy(results['img']).to(device=device)
        
        # (H, W, C) -> (C, H, W)
        if self.need_permute:
            img = img.permute(2, 0, 1)
         
        f_img = img
        # draw bboxes
        if results.get('gt_bboxes'):
            f_img = f_img.permute(1, 2, 0)
            f_img = mmcv.imshow_bboxes(
                img=f_img.cpu().numpy(),
                bboxes=results['gt_bboxes'].tensor.cpu().numpy(),
                show=False,
            )
            f_img = torch.from_numpy(f_img).to(device=device).permute(2, 0, 1)    
        
        if self.extend_mode == ExtendType.ZERO:
            zero_img = torch.zeros_like(img)
            cubemap = {
                'F': f_img,
                'B': zero_img,
                'L': zero_img,
                'R': zero_img,
                'U': zero_img,
                'D': zero_img,
            }    
        else:
            flip_img = img.flip(dims=(2,))
            cubemap = {
                'F': f_img,
                'B': img,
                'L': flip_img,
                'R': flip_img,
                'U': self._build_up(img),
                'D': self._build_down(img),
            }
            
        results['img'] = cube2equi(
            cubemap=cubemap,
            cube_format='dict',
            width=self.width,
            height=self.height,
            mode=mode if mode else self.mode,
            backend='native',
        )
        if self.need_permute:
            results['img'] = results['img'].permute(1, 2, 0)
        results['img_shape'] = results['img'].size()[:2]
        if not is_tensor:
            results['img'] = results['img'].cpu().numpy()
                   
        return results

    @staticmethod
    def _build_up(img: torch.Tensor) -> torch.Tensor:
        """Build up face of the cube."""
        half_w = img.size()[1] // 2
        mask = img.new_ones((half_w, half_w))
        mask = mask.triu(diagonal=1) + torch.diag(img.new_ones((half_w,))) / 2
        mask = torch.cat([mask, mask.flip(dims=(1,))], dim=1)
        mask = F.pad(mask, (0, 0, 0, half_w))
        ret = (img * mask).flip(dims=(1,))
        ret += ret.transpose(1, 2).clone()
        return ret + ret.flip(dims=(1, 2))

    @classmethod
    def _build_down(cls, img: torch.Tensor) -> torch.Tensor:
        """Build down face of the cube."""
        return cls._build_up(img.flip(dims=(1, 2)))
    