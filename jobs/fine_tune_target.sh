#!/bin/bash

root_dir='/dfs/scratch0/nicchiou/domain_adapt_cxr'
model_path=$root_dir'/results/midrc/resnet50_hidden-1024_source-IL_target-CA'
log_dir='jobs/logs/'

seed=(0 1 2 3 4)
train_state=('CA' 'IN' 'NC' 'TX')

for i in ${!seed[@]}; do
for j in ${!train_state[@]}; do

exp_dir='resnet50_hidden-1024_source-IL_target-'${train_state[$j]}'_nt-all'

model_fname=$model_path'/source_checkpoint_'${seed[$i]}'.pt'

echo $exp_dir

python ~/domain_adapt_cxr/approaches/midrc/train.py \
    --exp_dir $exp_dir \
    --log_dir $log_dir \
    --gpus 0 2 \
    --train_state ${train_state[$j]} \
    --test_state IL \
    --n_samples -1 \
    --domain target \
    --resnet resnet50 \
    --hidden_size 1024 \
    --load_pretrained $model_fname \
    --fine_tune_modules \
        resnet.conv1 resnet.bn1 \
        resnet.layer1 resnet.layer2 resnet.layer3 resnet.layer4 \
        resnet.fc linear \
    --epochs 50 \
    --lr 0.0003 \
    --batch_size 64 \
    --seed ${seed[$i]} \
    --verbose;

done
done
