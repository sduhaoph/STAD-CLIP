#!/bin/bash

python3 /home/hph/Desktop/LLM/LLM/LLM/LLM/TTHF/main.py \
    --train \
    --lr_clip 5e-6 \
    --wd 1e-4 \
    --epochs 100 \
    --batch_size 56 \
    --dataset DoTA \
    --freezen_clip true \
    --gpu_num 4 \
    --height 224 \
    --width 224 \
    --normal_class 1 \
    --eval_every 2000 \
    --base_model 'ViT-B-16' \
    --general \
    --fg \
    --other_method 'TDAFF_BASE' \
    --exp_name 'TDAFF_BASE_RN50'

