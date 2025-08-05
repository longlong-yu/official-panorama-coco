"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-04-17
@desc:   
"""
import json
import mmcv
import os
import sys
sys.path.append('.')
from typing import List
import torch
from panorama_coco.utils.spherical import SphereImageUtils
from panorama_coco.visualization.visualizer import BaseVisualizer


def main(
    classes: List[str],
    image_ids: List[int],
    src_dir: str,
    config: dict,
):
    with open(os.path.join(src_dir, config['ann_file']), 'r') as f:
        src_anns = json.load(f)
    
    category_ids = {item['id']: item['name'] for item in src_anns['categories'] if classes is None or item['name'] in (classes or [])}
    max_category = max(category_ids.keys())
    class_names = []
    for i in range(max_category):
        class_names.append(category_ids.get(i + 1, ''))
    
    src_dict = {}
    for meta in src_anns['images']:
        src_dict[meta['id']] = {
            'image': meta,
            'anns': []
        }  
    for ann in src_anns['annotations']:
        if ann['iscrowd']:
            continue
        src_dict[ann['image_id']]['anns'].append(ann)
        
    visualizer = BaseVisualizer(
         line_width=0.5,
    )
    colors = visualizer._get_palette(max_category * 3 + 4, mode='xkcd')
    for image_id in image_ids:
        if image_id not in src_dict:
            continue
        
        src_meta = src_dict[image_id]['image']
        src_anns = src_dict[image_id]['anns']
         
        image_path = os.path.join(src_dir, config['data_prefix'], src_meta['file_name'])
        image = mmcv.imread(image_path)
        visualizer.set_image(image)
        for ann in src_anns:
            if classes is not None and ann['category_id'] not in category_ids:
                continue
            
            # Draw original bounding boxes.
            bbox = [
                ann['bbox'][0],
                ann['bbox'][1],
                SphereImageUtils.angle2rad(ann['bbox'][2]), 
                SphereImageUtils.angle2rad(ann['bbox'][3]), 
                SphereImageUtils.angle2rad(ann['bbox'][4])
            ]
            bboxes = torch.tensor([bbox])
            positions = visualizer.draw_sboxes(
                bboxes=bboxes,
                erp_w=src_meta['width'],
                erp_h=src_meta['height'],
                colors=[colors[ann['category_id'] * 3 + 2]],
                # colors=[colors[30]]
            )
            visualizer.draw_labels(
                positions=positions,
                areas=bboxes[:, 2] * bboxes[:, 3] * (src_meta['width'] * src_meta['height'] / 32400),
                labels=[ann['category_id'] - 1],
                classes=class_names,
                colors=[colors[ann['category_id'] * 3 + 2]],
            )
            
        new_image = visualizer.get_image()
        mmcv.imwrite(img=new_image, file_path=f'local-data/local-output/pandora/{image_id}_src.jpg')     


if __name__ == '__main__':
    src_dir = 'data/PANDORA'
    configs = [
        {
            'ann_file': 'annotations/train.json',
            'data_prefix': 'images',
        },
        {
            'ann_file': 'annotations/test.json',
            'data_prefix': 'images', 
        },   
    ]
    
    classes = (
        'backpack', 'bed', 'book', 'bottle', 'bowl', 
        'chair', 'clock', 'cup', 'keyboard', 'microwave', 
        'mouse', 'oven', 'person', 'potted plant', 'refrigerator', 
        'sink', 'toilet', 'tv', 'vase', 'wine glass'
    )
    
    is_val = int(sys.argv[1])
    image_ids = [int(item) for item in sys.argv[2].split(',')]
    main(
        classes=None,
        image_ids=image_ids,
        src_dir=src_dir,
        config=configs[is_val],
    )
