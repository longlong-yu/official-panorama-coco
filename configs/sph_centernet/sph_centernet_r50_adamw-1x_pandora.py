"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-02-04
@desc:   
"""
_base_ = './sph_centernet_r101_adamw-1x_pandora.py'


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
    backbone=dict(
        depth=50,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet50',
            # checkpoint='local-data/saved_models/20240722_165714_r50_4×_60e_coco/epoch_60_backbone.pth',
        ),
    ),
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
work_dir = f'local-data/local-log/{datetime.now().strftime("%Y%m%d_%H%M%S")}_r50_1×_120e'
del datetime
