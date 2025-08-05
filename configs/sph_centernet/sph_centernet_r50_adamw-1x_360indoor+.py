"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-02-04
@desc:   
"""
_base_ = [
    '../_base_/datasets/360indoor+.py',
    '../_base_/schedules/schedule_1x_120e.py',
    './sph_centernet_r101_adamw.py'
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

train_cfg = dict(val_begin=110, val_interval=1)

# model settings
model = dict(
    backbone=dict(
        depth=50,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet50',
        ),
    ),
    bbox_head=dict(
        erp_shape={{_base_._erp_shape}},
        num_classes=_num_classes,
    ),
)

train_dataloader = dict(
    num_workers=2,
    # sampler=dict(type='mmdet.DefaultSampler', shuffle=False),
    dataset=dict(
        metainfo=_metainfo,
        limit_n=0,
    )
)
val_dataloader = dict(
    num_workers=2,
    dataset=dict(
        metainfo=_metainfo,
        test_mode=True,
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
work_dir = f'local-data/local-log/{datetime.now().strftime("%Y%m%d_%H%M%S")}_r50_1x_120e_360indoor+'
del datetime
