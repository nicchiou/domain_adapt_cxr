#!/bin/bash

res_dir='early_stop_auc/film_block-1234_bn-3_all/'
log_dir='jobs/logs/'
approach='FiLM_Target_ERM'

resnet='resnet152'
hidden_size=1024
block_replace='1 2 3 4'
bn_replace='3'

gpus='0 1 2 3'
seed=(0 1 2 3 4)
train_state=('CA' 'IN' 'TX')

for i in ${!seed[@]}; do
    for j in ${!train_state[@]}; do
        exp_dir=$resnet'_source-'${train_state[$j]}'_target-IL_ft-all_ns-all_nt-0'
        echo $exp_dir

        python ~/domain_adapt_cxr/approaches/midrc/train.py \
            --res_dir $res_dir \
            --exp_dir $exp_dir \
            --log_dir $log_dir \
            --approach $approach \
            --gpus $gpus \
            --train_state ${train_state[$j]} \
            --test_state IL \
            --n_samples -1 \
            --domain source \
            --resnet $resnet \
            --hidden_size $hidden_size \
            --film \
            --block_replace $block_replace \
            --bn_replace $bn_replace \
            --epochs 100 \
            --lr 0.001 \
            --batch_size 128 \
            --seed ${seed[$i]} \
            --early_stopping_metric auc \
            --min_epochs 10 \
            --verbose
    done
done