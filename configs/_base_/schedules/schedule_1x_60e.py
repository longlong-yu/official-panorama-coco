"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2023-12-20
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
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500
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
optim_wrapper = dict(optimizer=dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05
))

auto_scale_lr = dict(base_batch_size=16, enable=False)
