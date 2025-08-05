default_scope = 'panorama_coco'


default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=12, save_best='coco/bbox_mAP'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='VisualizationHook',
        train=dict(
            enabled=False,
            interval=500,
            max_n=0,
            per_n=1,
            channel_configs=dict(
                main=dict(
                    valid_names=({
                        'gt_hbox', 
                        # 'pred_hbox', 'gt_sbox', 'pred_sbox',
                        'gt_heatmap', 
                        # 'gt_mask',
                        'pred_heatmap_level0', 'pred_heatmap_level1',
                    }),
                ),
                origin=dict(
                    image=False,
                ),
                ws_erp=dict( 
                    image=False,
                    valid_names=set({
                        'gt_heatmap', 'gt_mask',
                        'pred_heatmap_level0', 'pred_heatmap_level1',
                        'pred_sbox_level0', 'pred_sbox_level1',
                    }),
                ),
                ss_erp=dict(
                    image=False,
                    valid_names=set({
                        'gt_sbox',
                        'pred_sbox_level0', 'pred_sbox_level1',
                    }),
                ),
            )
        ),
        val=dict(
            enabled=False,
            interval=500,
            max_n=0,
            per_n=1,
            channel_configs=dict(
                main=dict(
                    score_thr=0,
                    max_bbox_n=20,
                    valid_names=set({
                        # 'gt_hbox', 'pred_hbox', 
                        'gt_sbox', 'pred_sbox',
                        'pred_heatmap_level0', 'pred_heatmap_level1',
                    }),
                ),
            )
        ),
    )
)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=24),
    # mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='BaseVisualizer',
    name='visualizer',
    vis_backends=vis_backends, 
    line_width=0.5,
)
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
# load from which checkpoint
load_from = None
# whether to resume training from the loaded checkpoint
resume = False
# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)