"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2025-05-03
@desc:   
"""
_base_ = [
    '../_base_/datasets/360indoor+.py',
    '../_base_/schedules/schedule_sgd_1x_120e.py',
    './sph_retinanet_50_fpn.py'
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

box_type = 'sbox'
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
        num_classes=_num_classes,
        anchor_generator=dict(
            # box_type=box_type,
        ),
        bbox_coder=dict(
            type='DeltaXYWHASphBBoxCoder',
            target_means=[.0, .0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 1.0],
            box_type=box_type,
            # delta_ratio=100.0,
            # angle_mode='sin'
        ),
        reg_decoded_bbox=False,
    ),
    train_cfg=dict(
        assigner=dict(
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
        ),
    ),
    test_cfg=dict(
        # nms 之前的 topk bboxes
        nms_pre=200,
        score_thr=0.05,
        nms=dict(iou_threshold=0.5),
    )
)

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='mmdet.Resize', scale={{_base_._erp_shape}}, keep_ratio=True),
    dict(
        type='mmdet.LoadAnnotations',
        with_bbox=True,
        with_mask=False,
        poly2mask=False,
        box_type=box_type,
    ),
    dict(
        type='mmdet.RandomFlip',
        prob=0.5,
        direction=['horizontal', 'vertical', 'diagonal']
    ),
    dict(type='RandomRotateERP'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=(
            # default keys
            'img_id', 'img_path', 'ori_shape', 'img_shape',
            'instances'
        ),
    )
]

train_dataloader = dict(
    batch_size=16,
    num_workers=1,
    # sampler=dict(type='mmdet.DefaultSampler', shuffle=False),
    dataset=dict(
        metainfo=_metainfo,
        pipeline=train_pipeline,
        limit_n=0, 
    )
)

val_dataloader = dict(
    batch_size=16,
    num_workers=1,
    dataset=dict(
        metainfo=_metainfo,
        test_mode=True,
        limit_n=0,
    )
)
test_dataloader = val_dataloader

train_cfg = dict(val_interval=1, val_begin=110)

from datetime import datetime
work_dir = f'local-data/local-log/{datetime.now().strftime("%Y%m%d_%H%M%S")}_r50_retina-1x_120e_360indoor+'
del datetime
