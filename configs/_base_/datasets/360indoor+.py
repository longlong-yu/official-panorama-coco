"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2025-05-03
@desc:   
"""
# dataset settings
dataset_type = 'PanoramaDataset'
data_root = 'data'
backend_args = None

_erp_shape = (512, 1024)
_metainfo = dict(classes=(
    'airconditioner', 'backpack', 'bathtub', 'bed', 'board',
    'book', 'bottle', 'bowl', 'cabinet', 'chair', 'clock',
    'computer', 'cup', 'door', 'fan', 'fireplace', 'heater',
    'keyboard', 'light', 'microwave', 'mirror', 'mouse', 'oven',
    'person', 'phone', 'picture', 'potted plant', 'refrigerator',
    'sink', 'sofa', 'table', 'toilet', 'tv', 'vase', 'washer',
    'window', 'wine glass'
))
_num_classes = 37

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=_erp_shape, keep_ratio=True),
    dict(
        type='mmdet.LoadAnnotations',
        with_bbox=True,
        with_mask=False,
        poly2mask=False,
        box_type='sbox',
    ), 
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
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
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=_erp_shape, keep_ratio=True),
    # avoid bboxes being resized
    dict(
        type='mmdet.LoadAnnotations',
        with_bbox=True,
        with_mask=False,
        poly2mask=False,
        box_type='sbox',
    ),
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
    batch_size=12,
    num_workers=1,
    pin_memory=True,
    persistent_workers=True,
    sampler=dict(type='mmdet.DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        pipeline=train_pipeline,
        metainfo=_metainfo,
        data_root=data_root,
        ann_file='360-Indoor/annotations/train+.json',
        data_prefix=dict(img='360-Indoor/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        limit_n=0,
    )
)
val_dataloader = dict(
    batch_size=12,
    num_workers=1,
    pin_memory=True,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='mmdet.DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        pipeline=val_pipeline,
        metainfo=_metainfo,
        data_root=data_root,
        ann_file='360-Indoor/annotations/test+.json',
        data_prefix=dict(img='360-Indoor/images/'),
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=True),
        limit_n=0,
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='PandoraMetric',
    classwise=True,
)

test_evaluator = val_evaluator
