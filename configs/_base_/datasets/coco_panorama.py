"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-04-17
@desc:   
"""
# dataset settings
dataset_type = 'PanoramaDataset'
data_root = 'data'
backend_args = None

_erp_shape = (512, 1024)
_metainfo = dict(classes=(
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
))
_num_classes = 80

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
        ann_file='coco_panorama/annotations/instances_train2017.json',
        data_prefix=dict(img='coco_panorama/train2017/'),
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
        ann_file='coco_panorama/annotations/instances_val2017.json',
        data_prefix=dict(img='coco_panorama/val2017/'),
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
