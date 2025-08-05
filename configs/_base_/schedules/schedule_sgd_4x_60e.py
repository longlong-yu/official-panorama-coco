"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-09-01
@desc:   
"""

# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=60, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=10
    ),
    dict(
        type='MultiStepLR',
        begin=0,
        end=60,
        by_epoch=True,
        milestones=[40, 55],
        gamma=0.1
    )
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
    # clip_grad=None,
    clip_grad=dict(max_norm=15, norm_type=2, error_if_nonfinite=False)
)

auto_scale_lr = dict(base_batch_size=16 * 4, enable=False)
