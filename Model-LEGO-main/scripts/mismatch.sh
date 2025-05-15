export PYTHONPATH="${PYTHONPATH}:./Model-LEGO"



for k in 'seed7' 'seed44' 'seed123' 
do

python -m core.mismatch_test \
  --model_name 'vgg16' \
  --data_name cifar10 \
  --num_classes 10\
  --model_path './Model-LEGO/saved_disassembled_models/vgg16_cifar10_disassembled_'${k}'_0.pth' \
  --data_dir './Data_for_Lego/cifar10_images/test' \
  --classes 0
  
python -m core.mismatch_test \
  --model_name 'vgg16' \
  --data_name cifar10 \
  --num_classes 10 \
  --model_path './Model-LEGO/saved_disassembled_models/vgg16_cifar10_disassembled_'${k}'_3_9.pth' \
  --data_dir './Data_for_Lego/cifar10_images/test' \
  --classes {3..9}
  
python -m core.mismatch_test \
  --model_name 'vgg16' \
  --data_name cifar10 \
  --num_classes 10 \
  --model_path './Model-LEGO/saved_disassembled_models/vgg16_cifar10_disassembled_'${k}'_4_8.pth' \
  --data_dir './Data_for_Lego/cifar10_images/test' \
  --classes {4..8}
  
  python -m core.mismatch_test \
  --model_name 'vgg16' \
  --data_name cifar10 \
  --num_classes 10 \
  --model_path './Model-LEGO/saved_disassembled_models/vgg16_cifar10_disassembled_'${k}'_1_5.pth' \
  --data_dir './Data_for_Lego/cifar10_images/test' \
  --classes {1..5}
  
done 
