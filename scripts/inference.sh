#!/bin/bash
model=create_uni3d

clip_model="EVA02-E-14-plus" 
pretrained="/path/to/clip_model/open_clip_pytorch_model.bin" # or  "laion2b_s9b_b144k"  

pc_model="eva_giant_patch14_560.m30m_ft_in22k_in1k"

ckpt_path="/path/to/checkpoints/model.pt"

torchrun --nproc-per-node=1 main.py \
    --model $model \
    --batch-size 32 \
    --npoints 10000 \
    --num-group 512 \
    --group-size 64 \
    --pc-encoder-dim 512 \
    --clip-model $clip_model \
    --pretrained $pretrained \
    --pc-model $pc_model \
    --pc-feat-dim 1408 \
    --embed-dim 1024 \
    --validate_dataset_name modelnet40_openshape \
    --validate_dataset_name_lvis objaverse_lvis_openshape \
    --validate_dataset_name_scanobjnn scanobjnn_openshape \
    --evaluate_3d \
    --ckpt_path $ckpt_path \
