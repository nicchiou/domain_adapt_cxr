#!/bin/bash

root_dir='/dfs/scratch0/nicchiou/domain_adapt_cxr'
res_dir='early_stop_loss/film_block-4_bn-0123_all/'
log_dir='jobs/logs/'
approach='Source_ERM_FiLM'

resnet='resnet152'
hidden_size=1024
block_replace='4'
bn_replace='0 1 2 3'

gpus='0 1 2 5 9'
seed=(0 1 2 3 4)
test_state=('IN' 'TX')

for i in ${!seed[@]}; do
    exp_dir=$resnet'_source-IL_target-CA_ft-all_ns-all_nt-0'
    echo $exp_dir

    python ~/domain_adapt_cxr/approaches/midrc/train.py \
        --res_dir $res_dir \
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
        --film \
        --block_replace $block_replace \
        --bn_replace $bn_replace \
        --epochs 100 \
        --lr 0.001 \
        --batch_size 128 \
        --seed ${seed[$i]} \
        --early_stopping_metric loss \
        --min_epochs 10 \
        --verbose

    model_path=$root_dir'/results/midrc/'$res_dir$exp_dir
    model_fname=$model_path'/source_checkpoint_'${seed[$i]}'.pt'
    
    python ~/domain_adapt_cxr/approaches/midrc/record_statistics.py \
            --res_dir $res_dir \
            --exp_dir $exp_dir \
            --log_dir $log_dir \
            --approach $approach \
            --gpus $gpus \
            --state IL \
            --n_samples -1 \
            --resnet $resnet \
            --hidden_size $hidden_size \
            --film \
            --block_replace $block_replace \
            --bn_replace $bn_replace \
            --load_pretrained $model_fname \
            --batch_size 16 \
            --seed ${seed[$i]} \
            --verbose
    
    python ~/domain_adapt_cxr/approaches/midrc/record_statistics.py \
            --res_dir $res_dir \
            --exp_dir $exp_dir \
            --log_dir $log_dir \
            --approach $approach \
            --gpus $gpus \
            --state CA \
            --n_samples -1 \
            --resnet $resnet \
            --hidden_size $hidden_size \
            --film \
            --block_replace $block_replace \
            --bn_replace $bn_replace \
            --load_pretrained $model_fname \
            --batch_size 16 \
            --seed ${seed[$i]} \
            --verbose

    for j in ${!test_state[@]}; do
        exp_dir=$resnet'_source-IL_target-'${test_state[$j]}'_ft-all_ns-all_nt-0'
        echo $exp_dir

        python ~/domain_adapt_cxr/approaches/midrc/train.py \
            --res_dir $res_dir \
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
            --film \
            --block_replace $block_replace \
            --bn_replace $bn_replace \
            --load_pretrained $model_fname \
            --inference_only \
            --epochs 100 \
            --lr 0.001 \
            --batch_size 128 \
            --seed ${seed[$i]} \
            --early_stopping_metric loss \
            --min_epochs 10 \
            --verbose
        
        python ~/domain_adapt_cxr/approaches/midrc/record_statistics.py \
            --res_dir $res_dir \
            --exp_dir $exp_dir \
            --log_dir $log_dir \
            --approach $approach \
            --gpus $gpus \
            --state ${test_state[$j]} \
            --n_samples -1 \
            --resnet $resnet \
            --hidden_size $hidden_size \
            --film \
            --block_replace $block_replace \
            --bn_replace $bn_replace \
            --load_pretrained $model_fname \
            --batch_size 16 \
            --seed ${seed[$i]} \
            --verbose
    done
done