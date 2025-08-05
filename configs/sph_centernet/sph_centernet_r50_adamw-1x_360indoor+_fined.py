"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2025-05-03
@desc:   
"""
_base_ = './sph_centernet_r50_adamw-1x_360indoor+.py'


# load_from = 'local-data/local-log/20240722_154400_r50_4×_60e_coco/epoch_60.pth'
resume = False

_metainfo = dict(classes=(
    'backpack', 'bed', 'book', 'bottle', 'bowl', 'chair',
    'clock', 'cup', 'keyboard', 'microwave', 'mouse', 'oven',
    'person', 'potted plant', 'refrigerator', 'sink', 'toilet', 
    'tv', 'vase', 'wine glass'
))
_num_classes = 20
# _metainfo = {{_base_._metainfo}}
# _num_classes = {{_base_._num_classes}}

# model settings
model = dict(
    bbox_head=dict(
        erp_shape={{_base_._erp_shape}},
        num_classes=_num_classes,
    ),
)

train_dataloader = dict(
    num_workers=2,
    dataset=dict(
        metainfo=_metainfo,
    )
)
val_dataloader = dict(
    num_workers=2,
    dataset=dict(
        metainfo=_metainfo,
    )
)
test_dataloader = val_dataloader

train_cfg = dict(max_epochs=48, val_interval=1, val_begin=38)

# learning rate
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=48,
        by_epoch=True,
        milestones=[32, 44],
        gamma=0.1
    )
]

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='AdamW',
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.05
))

from datetime import datetime
work_dir = f'local-data/local-log/{datetime.now().strftime("%Y%m%d_%H%M%S")}_r50_1×_360indoor+_fined'
del datetime
