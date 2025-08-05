"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-09-01
@desc:   
"""
_base_ = [
    '../_base_/datasets/coco_panorama.py',
    '../_base_/schedules/schedule_sgd_4x_60e.py',
    './sph_retinanet_50_fpn.py'
]

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
    num_workers=4,
    # sampler=dict(type='mmdet.DefaultSampler', shuffle=False),
    dataset=dict(
        ann_file='coco_panorama_reflect/annotations/instances_train2017.json',
        data_prefix=dict(img='coco_panorama_reflect/train2017/'),
        
        metainfo=_metainfo,
        pipeline=train_pipeline,
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
        
        metainfo=_metainfo,
        test_mode=True,
        limit_n=0,
    )
)
test_dataloader = val_dataloader


from datetime import datetime
work_dir = f'local-data/local-log/{datetime.now().strftime("%Y%m%d_%H%M%S")}_r50_retina-4x_120e_coco'
del datetime