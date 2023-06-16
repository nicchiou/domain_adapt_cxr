#!/bin/bash

root_dir='/dfs/scratch0/nicchiou/domain_adapt_cxr'
res_dir='early_stop_loss/resnet/'
log_dir='jobs/logs/'
approach='Fine_Tune_ResNet'

resnet='resnet152'
hidden_size=1024
model_path=$root_dir'/results/midrc/'$res_dir$resnet'_source-IL_target-CA_ft-all_ns-all_nt-0'

gpus='0 2 3 4 5'
seed=(0 1 2 3 4)
train_state=('CA' 'IN' 'TX')
n_samples=(20 50 100 200 300)

for i in ${!seed[@]}; do
    for j in ${!train_state[@]}; do
        for k in ${!n_samples[@]}; do
            exp_dir=$resnet'_source-IL_target-'${train_state[$j]}'_ft-block1_ns-all_nt-'${n_samples[$k]}
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
                --load_pretrained $model_fname \
                --fine_tune_modules block1 \
                --epochs 100 \
                --lr 0.0003 \
                --batch_size 128 \
                --seed ${seed[$i]} \
                --early_stopping_metric loss \
                --verbose
            
            model_fname=$root_dir'/results/midrc/'$res_dir$exp_dir'/target_checkpoint_'${seed[$i]}'.pt'
            
            python ~/domain_adapt_cxr/approaches/midrc/record_statistics.py \
                --res_dir $res_dir \
                --exp_dir $exp_dir \
                --log_dir $log_dir \
                --approach $approach \
                --gpus $gpus \
                --state ${train_state[$j]} \
                --n_samples ${n_samples[$k]} \
                --domain target \
                --resnet $resnet \
                --hidden_size $hidden_size \
                --load_pretrained $model_fname \
                --batch_size 16 \
                --seed ${seed[$i]} \
                --verbose
        done

        exp_dir=$resnet'_source-IL_target-'${train_state[$j]}'_ft-block1_ns-all_nt-all'
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
            --load_pretrained $model_fname \
            --fine_tune_modules block1 \
            --epochs 100 \
            --lr 0.0003 \
            --batch_size 128 \
            --seed ${seed[$i]} \
            --early_stopping_metric loss \
            --verbose
        
        model_fname=$root_dir'/results/midrc/'$res_dir$exp_dir'/target_checkpoint_'${seed[$i]}'.pt'
            
        python ~/domain_adapt_cxr/approaches/midrc/record_statistics.py \
            --res_dir $res_dir \
            --exp_dir $exp_dir \
            --log_dir $log_dir \
            --approach $approach \
            --gpus $gpus \
            --state ${train_state[$j]} \
            --n_samples ${n_samples[$k]} \
            --domain target \
            --resnet $resnet \
            --hidden_size $hidden_size \
            --load_pretrained $model_fname \
            --batch_size 16 \
            --seed ${seed[$i]} \
            --verbose
    done
done