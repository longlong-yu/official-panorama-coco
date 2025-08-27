"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-01-25
@desc:   
"""
import os
from typing import Any, List, Tuple
from mmdet.datasets.api_wrappers import COCO
from mmengine.fileio import get_local_path
from panorama_coco.datasets.base import BaseDataset
from panorama_coco.registry import DATASETS
from panorama_coco.utils.assert_ import assert_with_msg
   
   
@DATASETS.register_module()
class CocoDataset(BaseDataset):
    """ 
    COCO fasion dataset.
    Args:
        bbox_mode: ('xyxy', 'cxcywh')
        filter_cfg: 
            - filter_empty_gt: bool, gt is not empty.
            - min_size: int, min width/height.
            - classes: List[str], select classes to load, None means all classes
    """
    METAINFO = {
        'classes':(
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        )
     }
    
    BBOX_MODE_XYXY = 'xyxy'
    BBOX_MODE_CXCYWH = 'cxcywh'
    
    def __init__(
        self, 
        *, 
        bbox_mode: str = BBOX_MODE_XYXY,
        **kwargs
    ) -> None:
        self.bbox_mode = bbox_mode
        self.cat_id_to_label = {}
        super().__init__(**kwargs)
    
    def load_annotations(self) -> Tuple[Any, List[str]]:
        with get_local_path(self.ann_file) as local_path:
            coco = COCO(local_path)
        return coco, self.metainfo['classes']
     
    def load_raw_data(self, annotations: Any):
        coco = annotations
        total_ann_ids = []
        for category_id, category in coco.cats.items():
            if category['name'] in self.metainfo['classes']:
                self.cat_id_to_label[category_id] = self.category_to_label[category['name']]
        
        for img_id in coco.get_img_ids():
            raw_img_info = coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            yield {
                'raw_ann_info': raw_ann_info,
                'raw_img_info': raw_img_info
            }

        assert_with_msg(
            statement=len(set(total_ann_ids)) == len(total_ann_ids),
            msg=f"Annotation ids in '{self.ann_file}' are not unique!" 
        )

    def parse_raw_data(self, raw_data_info: dict) -> dict:
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}
        img_path = os.path.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = os.path.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix
            )
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        instances = []
        data_info['instances'] = instances
        for ann in ann_info:
            if ann.get('ignore', False) or ann['category_id'] not in self.cat_id_to_label:
                continue
            
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            
            instance = {}
            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            
            # we should use mode xyxy to be consistent with hbox. 
            if self.bbox_mode == self.BBOX_MODE_XYXY:
                instance['bbox'] = [x1, y1, x1 + w, y1 + h]
            else:
                instance['bbox'] = ann['bbox']
             
            instance['bbox_label'] = self.cat_id_to_label[ann['category_id']]
            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']
            instances.append(instance)
        
        return data_info

    def get_cat_ids(self, idx: int) -> List[int]:
        """
        Get category ids by index. Dataset wrapped by ClassBalancedDataset
        must implement this method.

        The ``ClassBalancedDataset`` requires a subclass which implements this
        method.

        Args:
            idx (int): The index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """
        return [item['bbox_label'] for item in self.get_data_info(idx)['instances']]

    def filter_hook(self, data_info: dict) -> bool:
        if self.filter_cfg is not None:        
            min_size = self.filter_cfg.get('min_size', 0)        
            width = data_info['width']
            height = data_info['height']
            if min(width, height) < min_size:
                return False
            
            classes = self.filter_cfg.get('classes', [])
            if classes:
                labels = {self.category_to_label[item] for item in classes}
                instances = []
                for item in data_info['instances']:
                    if item['bbox_label'] in labels:
                        instances.append(item)
                data_info['instances'] = instances
            
            image_ids = self.filter_cfg.get('image_ids', [])
            if image_ids:
                if data_info['img_id'] not in image_ids:
                    return False
            
            filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
            if filter_empty_gt and not data_info['instances']:
                return False
        
        return super().filter_hook(data_info)
