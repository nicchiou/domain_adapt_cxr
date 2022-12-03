#!/bin/bash

res_dir='NC_ERM_hyperparam_search/early_stop_auc/'
log_dir='jobs/logs/'
approach='Target_ERM'

resnet='resnet50'
hidden_size=1024
lr=(0.01 0.006 0.003 0.001 0.0006 0.0003 0.0001 0.00006 0.00003 0.00001)
batch_size=(64 128)

gpus='0 1 2 3'
seed=(0 1 2 3 4)

for i in ${!seed[@]}; do
    for j in ${!lr[@]}; do
        for k in ${!batch_size[@]}; do
            exp_dir=$resnet'_lr-'${lr[$j]}'_batch-'${batch_size[$k]}'_source-IL_target-NC_ns-0_nt-all'
            echo $exp_dir

            python ~/domain_adapt_cxr/approaches/midrc/train.py \
                --root_dir $res_dir \
                --exp_dir $exp_dir \
                --log_dir $log_dir \
                --approach $approach \
                --gpus $gpus \
                --train_state NC \
                --test_state IL \
                --n_samples -1 \
                --domain source \
                --resnet $resnet \
                --hidden_size $hidden_size \
                --epochs 100 \
                --lr ${lr[$j]} \
                --batch_size ${batch_size[$k]} \
                --seed ${seed[$i]} \
                --early_stopping_metric auc \
                --min_epochs 10 \
                --verbose
        done
    done
done