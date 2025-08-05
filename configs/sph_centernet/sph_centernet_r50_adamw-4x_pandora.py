"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-07-23
@desc:   
"""
_base_ = './sph_centernet_r50_adamw-1x_pandora.py'


# _erp_scale = (960, 1920)
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
    bbox_head=dict(
        erp_shape={{_base_._erp_shape}},
        num_classes=_num_classes,
    ),
)

train_dataloader = dict(
    num_workers=4,
    dataset=dict(
        metainfo=_metainfo,
    )
)
val_dataloader = dict(
    num_workers=4,
    dataset=dict(
        metainfo=_metainfo,
    )
)
test_dataloader = val_dataloader

from datetime import datetime
work_dir = f'local-data/local-log/{datetime.now().strftime("%Y%m%d_%H%M%S")}_r50_4×_120e'
del datetime
