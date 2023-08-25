#!/usr/bin/env bash

#GPUs
gpus=0

#Set paths
checkpoint_root=/home/priscilla/Codes/ChangeFormer/checkpoints
vis_root=/home/priscilla/Codes/ChangeFormer/vis
data_name=LEVIR


img_size=256    
batch_size=8   
lr=0.0001
#lr=0.0000001         
max_epochs=200
embed_dim=256

net_G=ChangeFormerV6        #ChangeFormerV6 is the finalized verion

lr_policy=linear
optimizer=adamw                 #Choices: sgd (set lr to 0.01), adam, adamw
loss=ce                         #Choices: ce, fl (Focal Loss), miou
multi_scale_train=True
multi_scale_infer=False
shuffle_AB=False

#Initializing from pretrained weights
pretrain=./pretrained/pretrained_changeformer.pt
#pretrain=./checkpoints/CD_LEVIR_gfn_ChangeFormerV6_LEVIR_b8_lr0.00001_adamw_train_val_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256/last_ckpt.pt

#Train and Validation splits
split=train         #trainval
split_val=val      #test
project_name=CD_S2Looking_gfn_v2_${net_G}_${data_name}_b${batch_size}_lr${lr}_${optimizer}_${split}_${split_val}_${max_epochs}_${lr_policy}_${loss}_multi_train_${multi_scale_train}_multi_infer_${multi_scale_infer}_shuffle_AB_${shuffle_AB}_embed_dim_${embed_dim}

CUDA_VISIBLE_DEVICES=0 python main_cd.py --img_size ${img_size} --loss ${loss} --checkpoint_root ${checkpoint_root} --vis_root ${vis_root} --lr_policy ${lr_policy} --optimizer ${optimizer} --pretrain ${pretrain} --split ${split} --split_val ${split_val} --net_G ${net_G} --multi_scale_train ${multi_scale_train} --multi_scale_infer ${multi_scale_infer} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --shuffle_AB ${shuffle_AB} --data_name ${data_name}  --lr ${lr} --embed_dim ${embed_dim}
