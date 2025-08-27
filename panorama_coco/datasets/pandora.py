"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-03-23
@desc:   
"""
import os
from panorama_coco.datasets.coco import CocoDataset
from panorama_coco.registry import DATASETS
from panorama_coco.utils.spherical import SphereImageUtils


@DATASETS.register_module()
class PandoraDataset(CocoDataset):
    """ PANDORA dataset. """
    
    def parse_raw_data(self, raw_data_info: dict) -> dict:
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}
        img_path = os.path.join(self.data_prefix['img'], img_info['file_name'])
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

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
                SphereImageUtils.angle2rad(ann['bbox'][2]), 
                SphereImageUtils.angle2rad(ann['bbox'][3]), 
                SphereImageUtils.angle2rad(ann['bbox'][4])
            ]
            instance['area'] = ann['area'] 
            instance['bbox_label'] = self.cat_id_to_label[ann['category_id']]
        
        return data_info
   