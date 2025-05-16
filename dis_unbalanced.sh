#!/bin/bash
export PYTHONPATH=Model-LEGO-main

model_name='vgg16'
data_name='cifar10'
num_classes='10'
model_path='saved_models/VGG16_CIFAR10_seed7.pth'
data_dir='saved_datasets/cifar10_images/train'
data_dir_test='saved_datasets/cifar10_images/test'
dis_dir='saved_dis_models'
num_samples_unbalanced='163 170 81 60 111 84 123 177 129 166 '
disa_layers='-1'

sample_dir='saved_samples/cifar10_images_resnet_unbalanced_seed7'
mask_dir='saved_masks/masks_VGG16_CIFAR10_unbalanced_seed7/'

dis_name='VGG16_CIFAR10_seed7_unbalanced_disassembled_0.pth'
subset_dir='saved_sub_datasets/cifar10_images_seed7/test_0'

disa_labels='0'


#python3 Model-LEGO-MAIN/core/sample_select_unbalanced.py \
#  --model_name ${model_name} \
#  --data_name ${data_name} \
#  --num_classes ${num_classes} \
#  --model_path ${model_path}  \
#  --data_dir  ${data_dir} \
#  --save_dir ${sample_dir} \
#  --num_samples ${num_samples_unbalanced}
##
#python3 Model-LEGO-MAIN/core/relevant_feature_identifying.py \
#  --model_name ${model_name} \
#  --data_name ${data_name} \
#  --num_classes ${num_classes} \
#  --model_path  ${model_path}\
#  --data_dir ${sample_dir} \
#  --save_dir ${mask_dir} \
#
#python3 Model-LEGO-MAIN/core/model_disassemble.py \
#  --model_name ${model_name} \
#  --num_classes ${num_classes} \
#  --model_path ${model_path} \
#  --mask_dir ${mask_dir}'/masks' \
#  --save_dir ${dis_dir} \
#  --disa_layers ${disa_layers} \
#  --disa_labels ${disa_labels} \
#  --model_save_name ${dis_name}
#
#python3 Model-LEGO-main/dataset_splitter.py \
#   --source_dir ${data_dir_test} \
#   --target_dir ${subset_dir} \
#   --start_ind 0 \
#   --end_ind 0

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
#  --data_dir ${data_dir_test}

dis_name='VGG16_CIFAR100_seed7_unbalanced_disassembled_0_2.pth'
subset_dir='saved_sub_datasets/cifar100_images_seed7/test_0_2'

disa_labels='0 1 2'

#python3 Model-LEGO-MAIN/core/model_disassemble.py \
#  --model_name ${model_name} \
#  --num_classes ${num_classes} \
#  --model_path ${model_path} \
#  --mask_dir ${mask_dir}'masks' \
#  --save_dir ${dis_dir} \
#  --disa_layers ${disa_layers} \
#  --disa_labels ${disa_labels} \
#  --model_save_name ${dis_name}
#
#python3 Model-LEGO-main/dataset_splitter.py \
#   --source_dir ${data_dir_test} \
#   --target_dir ${subset_dir} \
#   --start_ind 0 \
#   --end_ind 2

python3  Model-LEGO-main/engines/test.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --num_classes ${num_classes} \
  --model_path ${dis_dir}'/'${dis_name} \
  --data_dir ${subset_dir}

dis_name='VGG16_CIFAR100_seed7_unbalanced_disassembled_3_9.pth'
subset_dir='saved_sub_datasets/cifar100_images_seed7/test_3_9'

disa_labels='3 4 5 6 7 8 9'

#python3 Model-LEGO-MAIN/core/model_disassemble.py \
#  --model_name ${model_name} \
#  --num_classes ${num_classes} \
#  --model_path ${model_path} \
#  --mask_dir ${mask_dir}'masks' \
#  --save_dir ${dis_dir} \
#  --disa_layers ${disa_layers} \
#  --disa_labels ${disa_labels} \
#  --model_save_name ${dis_name}
#
#python3 Model-LEGO-main/dataset_splitter.py \
#   --source_dir ${data_dir_test} \
#   --target_dir ${subset_dir} \
#   --start_ind 3 \
#   --end_ind 9

python3  Model-LEGO-main/engines/test.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --num_classes ${num_classes} \
  --model_path ${dis_dir}'/'${dis_name} \
  --data_dir ${subset_dir}

dis_name='VGG16_CIFAR100_seed7_unbalanced_disassembled_6.pth'
subset_dir='saved_sub_datasets/cifar100_images_seed7/test_6'

disa_labels='6'

#python3 Model-LEGO-MAIN/core/model_disassemble.py \
#  --model_name ${model_name} \
#  --num_classes ${num_classes} \
#  --model_path ${model_path} \
#  --mask_dir ${mask_dir}'masks' \
#  --save_dir ${dis_dir} \
#  --disa_layers ${disa_layers} \
#  --disa_labels ${disa_labels} \
#  --model_save_name ${dis_name}
#
#python3 Model-LEGO-main/dataset_splitter.py \
#   --source_dir ${data_dir_test} \
#   --target_dir ${subset_dir} \
#   --start_ind 6 \
#   --end_ind 6

python3  Model-LEGO-main/engines/test.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --num_classes ${num_classes} \
  --model_path ${dis_dir}'/'${dis_name} \
  --data_dir ${subset_dir}

dis_name='VGG16_CIFAR100_seed7_unbalanced_disassembled_1_5.pth'
subset_dir='saved_sub_datasets/cifar100_images_seed7/test_1_5'

disa_labels='1 2 3 4 5'

#python3 Model-LEGO-MAIN/core/model_disassemble.py \
#  --model_name ${model_name} \
#  --num_classes ${num_classes} \
#  --model_path ${model_path} \
#  --mask_dir ${mask_dir}'masks' \
#  --save_dir ${dis_dir} \
#  --disa_layers ${disa_layers} \
#  --disa_labels ${disa_labels} \
#  --model_save_name ${dis_name}
#
#python3 Model-LEGO-main/dataset_splitter.py \
#   --source_dir ${data_dir_test} \
#   --target_dir ${subset_dir} \
#   --start_ind 1 \
#   --end_ind 5

python3  Model-LEGO-main/engines/test.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --num_classes ${num_classes} \
  --model_path ${dis_dir}'/'${dis_name} \
  --data_dir ${subset_dir}

#dis_name='VGG16_CIFAR100_seed7_unbalanced_disassembled_79.pth'
#subset_dir='saved_sub_datasets/cifar100_images_seed7/test_79'
#
#disa_labels='79'
#
#python3 Model-LEGO-MAIN/core/model_disassemble.py \
#  --model_name ${model_name} \
#  --num_classes ${num_classes} \
#  --model_path ${model_path} \
#  --mask_dir ${mask_dir}'masks' \
#  --save_dir ${dis_dir} \
#  --disa_layers ${disa_layers} \
#  --disa_labels ${disa_labels} \
#  --model_save_name ${dis_name}
#
##python3 Model-LEGO-main/dataset_splitter.py \
##   --source_dir ${data_dir_test} \
##   --target_dir ${subset_dir} \
##   --start_ind 6 \
##   --end_ind 6
#
#python3  Model-LEGO-main/engines/test.py \
#  --model_name ${model_name} \
#  --data_name ${data_name} \
#  --num_classes ${num_classes} \
#  --model_path ${dis_dir}'/'${dis_name} \
#  --data_dir ${data_dir_test}

