#!/bin/bash
model=create_uni3d

clip_model="EVA02-E-14-plus" 
pretrained="/path/to/clip_model/open_clip_pytorch_model.bin" # or "laion2b_s9b_b144k"  
embed_dim=1024

pc_model="eva_giant_patch14_560.m30m_ft_in22k_in1k"
pretrained_pc="/path/to/init_model/model.safetensors"
pc_feat_dim=1408

pc_encoder_dim=512 

torchrun --nnodes=$WORLD_SIZE \
    --nproc-per-node=8 \
    main.py \
    --model $model \
    --pretrain_dataset_name ensembled_embedding \
    --npoints 10000 \
    --num-group 512 \
    --group-size 64 \
    --clip-model $clip_model \
    --pc-model $pc_model \
    --pretrained $pretrained \
    --pretrained-pc $pretrained_pc \
    --warmup 10000 \
    --batch-size=48 \
    --epochs 200 \
    --pc-feat-dim=$pc_feat_dim \
    --pc-encoder-dim=$pc_encoder_dim \
    --embed-dim=$embed_dim \
    --lr=1e-3 \
    --point-lr=1e-3 \
    --drop-path-rate=0.20 \
    --wd=0.1 \
    --point-wd=0.1 \
    --ld=1.0 \
    --point-ld=0.95 \
    --grad-clip-norm=5.0 \
    --smoothing=0. \
    --seed 4096 \
    --patch-dropout=0.5 \
    --optimizer="adamw" \
    --enable-deepspeed \
    --zero-stage=1 \
    --validate_dataset_name modelnet40_openshape \
    --validate_dataset_name_lvis objaverse_lvis_openshape \
    --validate_dataset_name_scanobjnn scanobjnn_openshape \
    --use-embed \
    --wandb \
    # --use_lvis \ 
    # whether to use objaverse dataset during pretraining