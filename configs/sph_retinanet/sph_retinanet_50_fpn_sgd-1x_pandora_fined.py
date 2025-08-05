"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-09-01
@desc:   
"""
_base_ = './sph_retinanet_50_fpn_sgd-1x_pandora.py'

load_from = None
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

box_type = 'sbox'
# model settings
model = dict(
    bbox_head=dict(
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

train_cfg = dict(max_epochs=48, val_interval=1)

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


from datetime import datetime
work_dir = f'local-data/local-log/{datetime.now().strftime("%Y%m%d_%H%M%S")}_r50_retina-1x_48e_pandora_fined'
del datetime
