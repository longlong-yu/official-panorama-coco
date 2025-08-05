#!/usr/bin/env bash
cd /root/apps/panorama-h2rbox
source activate
conda activate openmmlab
# PORT=29501 ./scripts/dist_train.sh configs/centernet/centernet_r50_adamw-2x_coco.py 2
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./scripts/dist_train.sh configs/sph_centernet/sph_centernet_r50_adamw-4x_coco.py 4 \
    --resume --work-dir='local-data/local-log/20240729_093425_r50_4×_60e_coco'