#!/usr/bin/env bash
cd /root/apps/panorama-h2rbox
source activate
conda activate openmmlab
# config:
# 'configs/sph_centernet/sph_centernet_r50_adamw-1x_360indoor.py'
# 'configs/sph_centernet/sph_centernet_r50_adamw-1x_pandora.py'
# model:
# 'local-data/local-log/20240722_154400_r50_4×_60e_coco/epoch_60.pth'
# 'local-data/local-log/20240724_100237_r50_1×_fined/epoch_12.pth'
python tools/test_model.py val_models  \
    'configs/sph_centernet/sph_centernet_r50_adamw-1x_pandora.py' \
    'local-data/local-log/20240722_154400_r50_4×_60e_coco/epoch_60.pth'
