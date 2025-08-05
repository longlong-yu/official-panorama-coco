#!/usr/bin/env bash
cd /root/apps/panorama-h2rbox
source activate
conda activate openmmlab
# configs/sph_centernet/sph_centernet_r50_adamw-1x_pandora_fined.py
# configs/sph_centernet/sph_centernet_r50_adamw-1x_pandora.py
# configs/sph_centernet/sph_centernet_r50_adamw-1x_360indoor.py
# configs/centernet/centernet_r50_adamw-1x_coco.py
python tools/train.py configs/centernet/centernet_r50_adamw-1x_coco.py
