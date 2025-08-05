"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-02-04
@desc:   
"""
_base_ = [
    '../_base_/datasets/coco_panorama.py',
    '../_base_/schedules/schedule_4x_60e.py',
    './sph_centernet_r101_adamw.py',
]


# _metainfo = dict(classes=(
#     'backpack', 'bed', 'book', 'bottle', 'bowl', 'chair',
#     'clock', 'cup', 'keyboard', 'microwave', 'mouse', 'oven',
#     'person', 'potted plant', 'refrigerator', 'sink', 'toilet', 
#     'tv', 'vase', 'wine glass'
# ))
# _num_classes = 20
_metainfo = {{_base_._metainfo}}
_num_classes = {{_base_._num_classes}}

train_cfg = dict(val_begin=50)

# model settings
model = dict(
    bbox_head=dict(
        erp_shape={{_base_._erp_shape}},
        num_classes=_num_classes,
    ),
)

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    # sampler=dict(type='mmdet.DefaultSampler', shuffle=False),
    dataset=dict(
        ann_file='coco_panorama_reflect/annotations/instances_train2017.json',
        data_prefix=dict(img='coco_panorama_reflect/train2017/'),

        metainfo=_metainfo,
        limit_n=0,
        # filter_cfg=dict(orientation='^FU_.*$'),
    )
)
val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(
        type='PandoraDataset',
        ann_file='PANDORA/annotations/test.json',
        data_prefix=dict(img='PANDORA/images/'),
        # ann_file='coco_panorama_reflect/annotations/r1_instances_val2017.json',
        # data_prefix=dict(img='coco_panorama_reflect/val2017/'),
        
        metainfo=_metainfo,
        limit_n=0,
        # filter_cfg=dict(classes=(
        #     'backpack', 'bed', 'book', 'bottle', 'bowl', 'chair',
        #     'clock', 'cup', 'keyboard', 'microwave', 'mouse', 'oven',
        #     'person', 'potted plant', 'refrigerator', 'sink', 'toilet', 
        #     'tv', 'vase', 'wine glass'
        # )),
    )
)
test_dataloader = val_dataloader


from datetime import datetime
work_dir = f'local-data/local-log/{datetime.now().strftime("%Y%m%d_%H%M%S")}_r101_4x_60e_coco'
del datetime
