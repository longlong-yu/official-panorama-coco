"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2024-02-04
@desc:   
"""
_base_ = './sph_centernet_r101_adamw-4x_coco.py'


# model settings
model = dict(
    backbone=dict(
        depth=50,
        init_cfg=dict(
            type='Pretrained', checkpoint='torchvision://resnet50',
        ),
    ),
)

from datetime import datetime
work_dir = f'local-data/local-log/{datetime.now().strftime("%Y%m%d_%H%M%S")}_r50_4×_60e_coco'
del datetime