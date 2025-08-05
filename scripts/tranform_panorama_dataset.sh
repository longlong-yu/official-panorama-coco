#!/usr/bin/env bash
cd /root/apps/panorama-h2rbox
source activate
conda activate openmmlab
python ./tools/tranform_panorama_dataset.py 8 0,1,2,3,4,5,6,7 zero 0
