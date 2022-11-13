#!/bin/bash

log_dir='jobs/logs/'

seed=(0 1 2 3 4)
test_state=('CA' 'IN' 'NC' 'TX')

for i in ${!seed[@]}; do
for j in ${!test_state[@]}; do

exp_dir='resnet50_hidden-1024_source-IL_target-'${test_state[$j]}

echo $exp_dir

python ~/domain_adapt_cxr/approaches/midrc/train.py \
    --exp_dir $exp_dir \
    --log_dir $log_dir \
    --gpus 1 3 \
    --train_state IL \
    --test_state ${test_state[$j]} \
    --n_samples -1 \
    --domain source \
    --resnet resnet50 \
    --hidden_size 1024 \
    --epochs 100 \
    --lr 0.001 \
    --batch_size 128 \
    --seed ${seed[$i]} \
    --verbose;

done
done
