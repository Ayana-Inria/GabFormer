#!/usr/bin/env bash

gpus=0

data_name=LEVIR  # Choices: LEVIR, WHU (set the path in data_config.py)
net_G=GabFormer
split=test
vis_root=/home/Codes/GabFormer/vis
project_name=CD_GabFormer_LEVIR_b8_lr0.0001_adamw_train_val_300_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256
checkpoints_root=/home/Codes/GabFormer/checkpoints
checkpoint_name=ep151.pt
img_size=256
embed_dim=256 #Make sure to change the embedding dim (best and default = 256)

CUDA_VISIBLE_DEVICES=0 python eval_cd.py --split ${split} --net_G ${net_G} --embed_dim ${embed_dim} --img_size ${img_size} --vis_root ${vis_root} --checkpoints_root ${checkpoints_root} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}


