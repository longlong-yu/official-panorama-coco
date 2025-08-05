"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-07-24
@desc:   
"""
_base_ = [
    '../_base_/datasets/indoor360.py',
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

# model settings
model = dict(
    rotates=[],
    backbone=dict(
        depth=50,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet50',
            # checkpoint='local-data/saved_models/20240723_095027_r50_2x_60e_coco_o/epoch_60_backbone.pth',
        ),
    ),
    bbox_head=dict(
        type='SphCenterNetHead',
        erp_shape={{_base_._erp_shape}},
        num_classes=_num_classes,
    ),
)

train_dataloader = dict(
    # sampler=dict(type='mmdet.DefaultSampler', shuffle=False),
    dataset=dict(
        metainfo=_metainfo,
        limit_n=0,
    )
)
val_dataloader = dict(
    dataset=dict(
        metainfo=_metainfo,
        test_mode=True,
        limit_n=0,
    )
)
test_dataloader = val_dataloader

default_hooks = dict(
    checkpoint=dict(interval=12),
    visualization=dict(
        type='VisualizationHook',
        train=dict(
            enabled=False,
            interval=10,
            max_n=0,
            per_n=1,
            channel_configs=dict(
                main=dict(
                    valid_names=({
                        'gt_sbox', 'gt_heatmap',
                        'pred_sbox_level0', 'pred_heatmap_level0'
                    }),
                ),
            )
        ),
        val=dict(
            enabled=False,
            interval=5,
            max_n=0,
            per_n=1,
            channel_configs=dict(
                main=dict(
                    score_thr=0,
                    max_bbox_n=20,
                    valid_names=set({
                        'gt_sbox', 'pred_sbox',
                        'pred_heatmap_level0',
                    }),
                ),
            )
        ),
    )
)

from datetime import datetime
work_dir = f'local-data/local-log/{datetime.now().strftime("%Y%m%d_%H%M%S")}_r50_1x_120e_360indoor'
del datetime