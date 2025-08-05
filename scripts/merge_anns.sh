#!/usr/bin/env bash
cd /root/apps/panorama-h2rbox
source activate
conda activate openmmlab
python ./tools/merge_anns.py
