import math
import mmcv
from mmdet.structures.mask import PolygonMasks
from mmrotate.datasets import ConvertMask2BoxType as MM_ConvertMask2BoxType, Rotate as MM_Rotate
import numpy as np
import torch
from panorama_coco.const.transforms import ExtendType
from panorama_coco.registry import TRANSFORMS
from panorama_coco.utils.spherical import ImageUtils


@TRANSFORMS.register_module()
class ConvertMask2BoxType(MM_ConvertMask2BoxType):
    """Convert masks in results to a certain box type.

    Required Keys:

    - ori_shape
    - gt_bboxes (BaseBoxes[torch.float32])
    - gt_masks (BitmapMasks | PolygonMasks)
    - instances (List[dict]) (optional)
    Modified Keys:
    - gt_bboxes
    - gt_masks
    - instances

    Args:
        box_type (str): The destination box type.
        keep_mask (bool): Whether to keep the ``gt_masks``.
            Defaults to False.
    """
    def transform(self, results: dict) -> dict:
        """The transform function."""
        assert 'gt_masks' in results.keys()
        masks = results['gt_masks']
        results['gt_bboxes'] = self.box_type_cls.from_instance_masks(masks)
        if not self.keep_mask:
            results.pop('gt_masks')

        # Modify results['instances'] for RotatedCocoMetric
        converted_instances = []
        for instance in results['instances']:
            # fix bug for coco data loader 
            # m = np.array(instance['mask'][0])
            if isinstance(instance['mask'], dict) or len(instance['mask']) == 0:
                continue
            
            m = np.array(instance['mask'][0])
            m = PolygonMasks(
                [[m]], 
                results['ori_shape'][1],
                results['ori_shape'][0]
            )
            instance['bbox'] = self.box_type_cls.from_instance_masks(m).tensor[0].numpy().tolist()
            if not self.keep_mask:
                instance.pop('mask')
                # instance.pop('gt_masks')
            converted_instances.append(instance)
        results['instances'] = converted_instances

        return results


@TRANSFORMS.register_module()
class Rotate(MM_Rotate):
    """
    Support padding for new blank areas.
    """
    def __init__(
        self,
        *args, 
        mode: str,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.mode = ExtendType.of(mode)
    
    def _transform_img(self, results: dict) -> None:
        """Rotate the image."""
        h, w = results['img'].shape[:2]
        r = math.ceil(math.sqrt(h * h + w * w))
        pad = (
            math.ceil((r - w) / 2),
            math.floor((r - w) / 2),
            math.ceil((r - h) / 2),
            math.floor((r - h) / 2)
        )
        img = ImageUtils.bias_pad(
            input=results['img'],
            pad=pad,
            extend_type=self.mode,
        )
        img = mmcv.imrotate(
            img,
            self.rotate_angle,
            border_value=self.img_border_value,
            interpolation=self.interpolation
        )     
        results['img'] = img[pad[2]:pad[2] + h, pad[0]:pad[0] + w, :]
