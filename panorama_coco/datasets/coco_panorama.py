"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-04-17
@desc:   
"""
import os
import re
from panorama_coco.datasets.coco import CocoDataset
from panorama_coco.registry import DATASETS


@DATASETS.register_module()
class PanoramaDataset(CocoDataset):
    """ PANDORA dataset. """
    def __init__(
        self, 
        **kwargs
    ) -> None:
        orientation = kwargs.get('filter_cfg', {}).get('orientation', '')
        if orientation:
            kwargs['filter_cfg']['orientation'] = re.compile(orientation)
        super().__init__(**kwargs)
        
    
    def parse_raw_data(self, raw_data_info: dict) -> dict:
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}
        img_path = os.path.join(self.data_prefix['img'], img_info['file_name'])
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']
        if 'origin_h' in img_info:
            data_info['origin_h'] = img_info['origin_h']
            data_info['origin_w'] = img_info['origin_w']
        
        instances = []
        data_info['instances'] = instances
        for ann in ann_info:
            if ann.get('ignore', False) or ann['category_id'] not in self.cat_id_to_label:
                continue
            
            if ann['area'] <= 0:
                continue
            
            instance = {}
            instances.append(instance)
            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            
            instance['bbox'] = [
                ann['bbox'][0], 
                ann['bbox'][1],
                ann['bbox'][2], 
                ann['bbox'][3], 
                ann['bbox'][4]
            ]
            instance['area'] = ann['area'] 
            instance['bbox_label'] = self.cat_id_to_label[ann['category_id']]
            if 'orientation' in ann:
                instance['orientation'] = ann['orientation']
            
        
        return data_info
   
    def filter_hook(self, data_info: dict) -> bool:
        if not super().filter_hook(data_info):
            return False
        
        if self.filter_cfg:
            # Support filtering based on keys and prefixes.
            # orientation:[FBLR][UD] or [UD] + _[LRUD]*n
            orientation = self.filter_cfg.get('orientation', '')
            if orientation:
                data_info['instances'] = [
                    item for item in data_info['instances'] if orientation.match(item['orientation']) 
                ]
                return not not data_info['instances'] 
        
        return True
   