#!/bin/bash
export PYTHONPATH=/nfs3/hjc/projects/cnnlego/code
export CUDA_VISIBLE_DEVICES=0
result_path='/nfs3/hjc/projects/cnnlego/output'
#----------------------------------------
exp_name='lenet_cifar10_base'
#----------------------------------------
model_name='lenet'
#----------------------------------------
num_classes=10
#----------------------------------------
model_path=${result_path}'/'${exp_name}'/models/model_ori.pth'
mask_dir=${result_path}'/'${exp_name}'/contributions/masks'
save_dir=${result_path}'/'${exp_name}'/models'
#----------------------------------------
disa_layers='-1'
disa_labels='3 4'

python core/model_disassemble.py \
  --model_name ${model_name} \
  --num_classes ${num_classes} \
  --model_path ${model_path} \
  --mask_dir ${mask_dir} \
  --save_dir ${save_dir} \
  --disa_layers ${disa_layers} \
  --disa_labels ${disa_labels}