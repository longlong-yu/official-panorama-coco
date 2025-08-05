
import json
import os
import sys
sys.path.append('.')
from typing import Tuple
from panorama_coco.const.transforms import ExtendType


def main(classes: Tuple, config: dict, src_dir: str):
    ret = {}
    all_keys = ['F*', 'F', 'FP', 'FBLR*', 'ALL']
    for key in all_keys:
        ret[key] = {
            'image_ids': set({}),
            'amount': 0,
        }
    
    with open(os.path.join(src_dir, config['ann_file']), 'r') as f:
        ann_info = json.load(f)
    category_ids = {item['id']: item['name'] for item in ann_info['categories'] if classes is None or item['name'] in (classes or [])}
    for ann in ann_info['annotations']:
        if ann['category_id'] not in category_ids:
            continue

        valid_keys = ['ALL']
        if 'orientation' in ann:
            if ann['orientation'] == 'F_':
                valid_keys.append('F*')
            if ann['orientation'].startswith('F_'):
                valid_keys.append('F')
            if ann['orientation'].startswith('F'):
                valid_keys.append('FP')
            if ann['orientation'] in ['F_', 'B_', 'L_', 'R_']:
                valid_keys.append('FBLR*')
        
        for key in valid_keys:
            ret[key]['amount'] += 1
            ret[key]['image_ids'].add(ann['image_id'])
    for key in all_keys:
        ret[key] = {
            'amount': ret[key]['amount'],
            'image_amount': len(ret[key]['image_ids'])
        }
    return ret


if __name__ == '__main__':
    src_dir = 'data/coco'
    extend_type = ExtendType.CIRCULAR
    # if len(sys.argv) > 1:
    #     extend_type = ExtendType.of(sys.argv[1].strip())
    random_faces = 0
    src_dir = f'data/coco_panorama_{extend_type}'
    # src_dir = 'data/360-Indoor'
        
    configs = [
        {
            'ann_file': 'annotations/instances_train2017.json',
            'data_prefix': 'train2017',
        },
        {
            'ann_file': 'annotations/instances_val2017.json',
            'data_prefix': 'val2017', 
        },
        # PANDROA & 360-Indoor
        # {
        #     'ann_file': 'annotations/train.json',
        #     'data_prefix': 'images',
        # },
        # {
        #     'ann_file': 'annotations/test.json',
        #     'data_prefix': 'images', 
        # },   
    ]
    classes = (
        'backpack', 'bed', 'book', 'bottle', 'bowl', 
        'chair', 'clock', 'cup', 'keyboard', 'microwave', 
        'mouse', 'oven', 'person', 'potted plant', 'refrigerator', 
        'sink', 'toilet', 'tv', 'vase', 'wine glass'
    )
    
    for config in configs:
        print(f'****{config["data_prefix"]}****')
        ret = main(
            # classes=classes,
            classes=None,
            config=config,
            src_dir=src_dir
        )
        for k, v in ret.items():
            print(f'{k}: labels {v["amount"]}, images {v["image_amount"]}')
