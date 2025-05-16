#!/bin/bash
export PYTHONPATH=Model-LEGO-main

model_name='vgg16'
data_name='cifar10'
num_classes='10'

model_path='saved_models/VGG16_CIFAR10_seed7.pth'
data_dir='saved_datasets/cifar10_images/train'
data_dir_test='saved_datasets/cifar10_images/test'
sample_dir='saved_samples/cifar10_images_resnet_seed7'
mask_dir='saved_masks/masks_VGG16_CIFAR10_seed7/'
dis_dir='saved_dis_models'
dis_name='VGG16_CIFAR10_seed7_disassembled_0_2.pth'
subset_dir='saved_sub_datasets/cifar10_images_seed7/test_0_2'

disa_layers='-1'
disa_labels='0 1 2'

#python3 Model-LEGO-MAIN/core/sample_select.py \
#  --model_name ${model_name} \
#  --data_name ${data_name} \
#  --num_classes ${num_classes} \
#  --model_path ${model_path}  \
#  --data_dir  ${data_dir} \
#  --save_dir ${sample_dir} \
#  --num_samples 50
#
#python3 Model-LEGO-MAIN/core/relevant_feature_identifying.py \
#  --model_name ${model_name} \
#  --data_name ${data_name} \
#  --num_classes ${num_classes} \
#  --model_path  ${model_path}\
#  --data_dir ${sample_dir} \
#  --save_dir ${mask_dir} \
#
python3 Model-LEGO-MAIN/core/model_disassemble.py \
  --model_name ${model_name} \
  --num_classes ${num_classes} \
  --model_path ${model_path} \
  --mask_dir ${mask_dir}'/masks' \
  --save_dir ${dis_dir} \
  --disa_layers ${disa_layers} \
  --disa_labels ${disa_labels} \
  --model_save_name ${dis_name}

python3 Model-LEGO-main/dataset_splitter.py \
   --source_dir ${data_dir_test} \
   --target_dir ${subset_dir} \
   --start_ind 0 \
   --end_ind 2

python3  Model-LEGO-main/engines/test.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --num_classes ${num_classes} \
  --model_path ${dis_dir}'/'${dis_name} \
  --data_dir ${subset_dir}


#python3  Model-LEGO-main/engines/test.py \
#  --model_name ${model_name} \
#  --data_name ${data_name} \
#  --num_classes ${num_classes} \
#  --model_path ${model_path} \
#  --data_dir ${subset_dir}

