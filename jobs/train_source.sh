#!/bin/bash

root_dir='/dfs/scratch0/nicchiou/domain_adapt_cxr'
res_dir='early_stop_auc/resnet/'
log_dir='jobs/logs/'
approach='Source_ERM'

resnet='resnet152'
hidden_size=1024

gpus='0 1 2 3'
seed=(0 1 2 3 4)
test_state=('IN' 'NC' 'TX')

for i in ${!seed[@]}; do
    exp_dir=$resnet'_source-IL_target-CA_ns-all_nt-0'
    echo $exp_dir

    python ~/domain_adapt_cxr/approaches/midrc/train.py \
        --root_dir $res_dir \
        --exp_dir $exp_dir \
        --log_dir $log_dir \
        --approach $approach \
        --gpus $gpus \
        --train_state IL \
        --test_state CA \
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
        --verbose

    model_path=$root_dir'/results/midrc/'$res_dir$exp_dir
    model_fname=$model_path'/source_checkpoint_'${seed[$i]}'.pt'

    for j in ${!test_state[@]}; do
        exp_dir=$resnet'_source-IL_target-'${test_state[$j]}'_ns-all_nt-0'
        echo $exp_dir

        python ~/domain_adapt_cxr/approaches/midrc/train.py \
            --root_dir $res_dir \
            --exp_dir $exp_dir \
            --log_dir $log_dir \
            --approach $approach \
            --gpus $gpus \
            --train_state IL \
            --test_state ${test_state[$j]} \
            --n_samples -1 \
            --domain source \
            --resnet $resnet \
            --hidden_size $hidden_size \
            --load_pretrained $model_fname \
            --inference_only \
            --epochs 100 \
            --lr 0.001 \
            --batch_size 128 \
            --seed ${seed[$i]} \
            --early_stopping_metric auc \
            --min_epochs 10 \
            --verbose
    done
done