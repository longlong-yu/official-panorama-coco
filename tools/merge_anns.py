"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-04-17
@desc:   
"""
import json
import math
import os
import sys
sys.path.append('.')
from typing import List


def main(
    dest_dir: str,
    src_dir: str,
    configs: List[dict],  
):
    for config in configs:
        ret = {
            'images': {},
            'annotations': {},
            'categories': []
        }
        with open(os.path.join(src_dir, config['ann_file']), 'r') as f:
            anns = json.load(f)
        ret['categories'] = anns['categories']
        
        ret_file = os.path.join(dest_dir, config['ann_file'])
        ret_fn = os.path.basename(config['ann_file'])
        ret_dir = os.path.dirname(ret_file)
        
        idx = 1
        for fn in os.listdir(ret_dir):
            fn = os.path.basename(fn)
            if fn.find(ret_fn) <= 0:
                continue
            is_image = fn.startswith('image_')
            is_ann = fn.startswith('ann_')
            if is_image or is_ann:
                with open(os.path.join(ret_dir, fn), 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        if not line:
                            continue
                        tmp = json.loads(line)
                        if is_image:
                            if 'face_anns' in tmp:
                                del tmp['face_anns']
                            ret['images'][tmp['id']] = tmp
                            
                        else:
                            if tmp['area'] == 0 or math.isnan(tmp['area']) or line.find('NaN') >=0 or line.find('nan') >= 0:
                                continue
                            
                            tmp['id'] = idx
                            idx += 1
                            ret['annotations'][f"{tmp['origin_id']}_{tmp['image_id']}_{tmp['orientation']}"] = tmp
      
        ret['images'] = list(ret['images'].values())
        ret['annotations'] = list(ret['annotations'].values())
        with open(ret_file, 'w') as f:
            json.dump(ret, f)


if __name__ == '__main__':
    dest_dir = 'data/coco_panorama_zero'
    src_dir = 'data/coco'
    configs = [
        {
            'ann_file': 'annotations/instances_train2017.json',
            'data_prefix': 'train2017',
        },
        {
            'ann_file': 'annotations/instances_val2017.json',
            'data_prefix': 'val2017', 
        },   
    ]
    main(
        dest_dir=dest_dir,
        src_dir=src_dir,
        configs=configs,
    )
