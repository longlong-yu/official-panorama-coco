"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-08-04
@desc:   
"""
_base_ = '../_base_/default_runtime.py'

custom_imports = dict(
    imports=[
        'mmrotate.visualization', 'mmrotate.models',
    ], 
    allow_failed_imports=False
)

box_version = 5
box_type = 'shbox'
model = dict(
    type='SphRetinaNet',
    # rotates=[(0., 0., 0.),(0., 0., 0.25),(0., 0., 0.5), (0., 0., -0.25), (0., 0.25, 0.), (0., -0.25, 0.)],
    # rotates=[(0., 0., 0.)],
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=4,
        upsample_cfg=dict(mode='nearest'),
        # upsample_cfg=dict(mode='bilinear'),
    ),
    bbox_head=dict(
        type='SphRetinaHead',
        with_erp_weights=False,
        predict_rescale=False,
        # num_classes=_num_classes,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='SphAnchorGenerator',
            box_version=box_version,
            box_formator='sph2pix',
            box_type=box_type,

            # octave_base_scale=4,
            # octave_base_scale=2, 
            # scales_per_octave=4,
            # 角度范围：[45, 90], [60, 120], [75, 150]
            # scales=[0.5, 1.2, 2.8],
            # scales=[1.414, 1.885, 2.357],
            # scales=[2.828, 3.770, 4.714],
            # 30, 45, 60, 75
            # scales=[1.885, 2.828, 3.770, 4.714],
            # 45, 55, 65, 75
            scales=[2.828, 3.457, 4.086, 4.714],
            ratios=[0.5, 1.0, 2.0],
            # strides=[8, 16, 32, 64, 128],
            strides=[8, 16, 32, 64]
        ),
        bbox_coder=dict(
            type='DeltaXYWHSphBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
            box_type=box_type,
        ),
        reg_decoded_bbox=True,
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=10.0
        ),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=1.0)
    ),
    train_cfg=dict(
        assigner=dict(
            type='mmdet.MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(
                type='SphOverlaps2D',
                backend='xinyuan',
                box_version=box_version
            )
        ),
        sampler=dict(type='mmdet.PseudoSampler'),  # Focal loss should use PseudoSampler
        allowed_border=-1,
        pos_weight=-1,
        debug=False
    ),
    test_cfg=dict(
        nms_pre=1000,
        local_maximum_kernel=3,
        min_bbox_size=0,
        score_thr=0.05,
        with_nms=True,
        # if necessary to eval in header
        need_eval=True,
        # !Note: set split_thr = 0 and class_agnostic = True to make different 
        # !Note: label run nms separately meanwhile keep original box position unchanged.
        nms=dict(
            type='sph2pob_nms', 
            split_thr=0,
            class_agnostic=True,
            max_num=0,
            iou_threshold=0.5,
            iou_calculator='sph2pob_efficient_iou',
        ),
        max_per_img=100
    )
)

default_hooks = dict(
    visualization=dict(
        type='VisualizationHook',
        train=dict(
            enabled=False,
            interval=20,
            max_n=0,
            per_n=1,
            channel_configs=dict(
                main=dict(
                    # channel_reduction='select_max',
                    max_bbox_n=50,
                    score_thr=-1,
                    valid_names=({
                        'gt_sbox', 
                        'gt_sbox_all', 
                        'gt_sbox_0', 'gt_sbox_1', 'gt_sbox_2', 'gt_sbox_3', 'gt_sbox_4',
                        'pred_sbox', 'pred_sbox_0', 'pred_sbox_1', 'pred_sbox_2', 'pred_sbox_3', 'pred_sbox_4',
                        'gt_heatmap_0', 'gt_heatmap_1', 'gt_heatmap_2', 'gt_heatmap_3', 'gt_heatmap_4',
                        'pred_heatmap_0', 'pred_heatmap_1', 'pred_heatmap_2', 'pred_heatmap_3', 'pred_heatmap_4',
                    }),
                ),
            )
        ),
        val=dict(
            enabled=False,
            interval=1,
            max_n=0,
            per_n=1,
            channel_configs=dict(
                main=dict(
                    score_thr=0,
                    max_bbox_n=0,
                    valid_names=set({
                        'gt_sbox', 'pred_sbox',
                    }),
                ),
            )
        ),
    )
)
