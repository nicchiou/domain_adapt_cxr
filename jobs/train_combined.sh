#!/bin/bash

root_dir='/dfs/scratch0/nicchiou/domain_adapt_cxr'
res_dir='early_stop_auc/combined/'
log_dir='jobs/logs/'
approach='Concat_ERM'

resnet='resnet152'
hidden_size=1024

gpus='0 1 2 3'
seed=(0 1 2 3 4)
test_state=('CA' 'IN' 'NC' 'TX')

for i in ${!seed[@]}; do
    for j in ${!test_state[@]}; do

    exp_dir=$resnet'_combined-IL-'${test_state[$j]}'_ns-all_nt-all'

    echo $exp_dir

    python ~/domain_adapt_cxr/approaches/midrc/train_combined.py \
        --root_dir $res_dir \
        --exp_dir $exp_dir \
        --log_dir $log_dir \
        --approach $approach \
        --gpus $gpus \
        --states IL ${test_state[$j]} \
        --n_samples -1 \
        --domain source \
        --resnet $resnet \
        --hidden_size $hidden_size \
        --epochs 100 \
        --lr 0.001 \
        --batch_size 128 \
        --seed ${seed[$i]} \
        --early_stopping_metric auc \
        --min_epochs 10 \
        --verbose;

    done
done
