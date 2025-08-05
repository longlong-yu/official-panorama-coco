"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-02-04
@desc:   
"""
_base_ = '../_base_/default_runtime.py'

custom_imports = dict(
    imports=[
        'mmrotate.visualization', 'mmrotate.models',
    ], 
    allow_failed_imports=False
)

# model settings
model = dict(
    type='SphCenterNet',
    # rotates=[(0., 0., 0.),(0., 0., 0.25),(0., 0., 0.5), (0., 0., -0.25), (0., 0.25, 0.), (0., -0.25, 0.)],
    # rotates=[(0., 0., 0.),0., 0.25, 0.],
    rotates=[(0., 0., 0.)],
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        # pad_size_divisor=1,
        boxtype2tensor=False
    ),
    backbone=dict(
        type='mmdet.ResNet',
        depth=101,
        num_stages=4, 
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='torchvision://resnet101',
        ),
    ),
    neck=dict(
        type='mmdet.CTResNetNeck',
        in_channels=2048,
        num_deconv_filters=(1024, 512, 256),
        num_deconv_kernels=(4, 4, 4),
        use_dcn=True
    ),
    bbox_head=dict(
        type='SphCenterNetHeadV2',
        # erp_shape={{_base_._erp_shape}},
        # num_classes={{_base_._num_classes}},
        in_channels=256,
        feat_channels=256,
        
        scale_angle=True,
        
        with_erp_ct_weight=False,
        auto_ajust=False,
        
        with_iou_loss=0,
        loss_iou=dict(type='TangentIouLoss', mode='linear', loss_weight=1.),
    ),
    test_cfg=dict(
        topk=100,
        local_maximum_kernel=3,
        score_threshold=0,
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
        max_num=100,
    ),
)

default_hooks = dict(
    visualization=dict(
        type='VisualizationHook',
        train=dict(
            enabled=False,
            interval=1,
            max_n=0,
            per_n=1,
            channel_configs=dict(
                main=dict(
                    valid_names=({
                        'gt_sbox', 'pred_sbox',
                        'gt_heatmap', 
                        'pred_heatmap_level0',
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
