#!/bin/bash

root_dir='/dfs/scratch0/nicchiou/domain_adapt_cxr'
res_dir='early_stop_auc/film_block-1234_bn-3_all/'
log_dir='jobs/logs/'
approach='Fine_Tune_FiLM'

resnet='resnet152'
hidden_size=1024
block_replace='1 2 3 4'
bn_replace='3'
model_path=$root_dir'/results/midrc/'$res_dir$resnet'_source-IL_target-CA_ft-all_ns-all_nt-0'

gpus='0 1 2 3'
seed=(0 1 2 3 4)
train_state=('CA' 'IN' 'TX')
n_samples=(20 50 100 200 300)

for i in ${!seed[@]}; do
    for j in ${!train_state[@]}; do
        for k in ${!n_samples[@]}; do
            exp_dir=$resnet'_source-IL_target-'${train_state[$j]}'_ft-all_ns-all_nt-'${n_samples[$k]}
            model_fname=$model_path'/source_checkpoint_'${seed[$i]}'.pt'
            echo $exp_dir

            python ~/domain_adapt_cxr/approaches/midrc/train.py \
                --res_dir $res_dir \
                --exp_dir $exp_dir \
                --log_dir $log_dir \
                --approach $approach \
                --gpus $gpus \
                --train_state ${train_state[$j]} \
                --test_state IL \
                --n_samples ${n_samples[$k]} \
                --domain target \
                --resnet $resnet \
                --hidden_size $hidden_size \
                --film \
                --block_replace $block_replace \
                --bn_replace $bn_replace \
                --load_pretrained $model_fname \
                --fine_tune_modules all \
                --epochs 100 \
                --lr 0.0003 \
                --batch_size 128 \
                --seed ${seed[$i]} \
                --early_stopping_metric auc \
                --verbose
        done

    exp_dir=$resnet'_source-IL_target-'${train_state[$j]}'_ft-all_ns-all_nt-all'
    model_fname=$model_path'/source_checkpoint_'${seed[$i]}'.pt'
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
        --domain target \
        --resnet $resnet \
        --hidden_size $hidden_size \
        --film \
        --block_replace $block_replace \
        --bn_replace $bn_replace \
        --load_pretrained $model_fname \
        --fine_tune_modules all \
        --epochs 100 \
        --lr 0.0003 \
        --batch_size 128 \
        --seed ${seed[$i]} \
        --early_stopping_metric auc \
        --verbose
    done
done