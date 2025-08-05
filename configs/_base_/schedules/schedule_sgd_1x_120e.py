"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-08-04
@desc:   
"""

# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=120, val_interval=4)
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
        end=120,
        by_epoch=True,
        milestones=[80, 110],
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

auto_scale_lr = dict(base_batch_size=16, enable=False)
