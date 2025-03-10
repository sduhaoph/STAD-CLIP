#!/bin/bash

python3 /home/haoph/Desktop/LLM/TTHF/main_box_new_chat.py \
    --evaluate \
    --lr_clip 5e-6 \
    --wd 1e-4 \
    --epochs 100 \
    --batch_size 128 \
    --dataset DoTA \
    --freezen_clip true \
    --gpu_num 4 \
    --height 224 \
    --width 224 \
    --normal_class 1 \
    --eval_every 1000 \
    --base_model 'ViT-B-16' \
    --general \
    --fg \
    --resume_model_path /home/haoph/Desktop/LLM/LLM/TTHF/checkpoint/87.01/best.pth \
    --other_method 'TDAFF_BASE' \
    --exp_name 'TDAFF_BASE_RN50'

