#!/bin/sh

src_batch_size=16
tgt_batch_size=16
n_target_samples=(20 50 100 200 500 1000)
train_seeds=(8 16 31)

for i in ${!n_target_samples[@]}; do
for value in {0..2}; do
  python ./train_classifier_add_film.py \
  --early_stop \
  --n_target_samples ${n_target_samples[$i]} \
  --source_batch_size $src_batch_size \
  --target_batch_size $tgt_batch_size \
  --exp_dir add_film_sparse_s_20k_t_${n_target_samples[$i]} \
  --iter_idx $value \
  --train_seed ${train_seeds[$value]} \
  --load_trained_model \
  --model_path experiments/new_baseline/fixed_batch_size/baseline_s_20k_t_20/source_checkpoint_$value.pt \
  --freeze \
  --fine_tune_layers resnet_fc linear \
  --film \
  --add_in_film \
  --replace_BN sparse;
  python ./train_classifier_add_film.py \
  --early_stop \
  --n_target_samples ${n_target_samples[$i]} \
  --source_batch_size $src_batch_size \
  --target_batch_size $tgt_batch_size \
  --exp_dir add_film_all_s_20k_t_${n_target_samples[$i]} \
  --iter_idx $value \
  --train_seed ${train_seeds[$value]} \
  --load_trained_model \
  --model_path experiments/new_baseline/fixed_batch_size/baseline_s_20k_t_20/source_checkpoint_$value.pt \
  --freeze \
  --fine_tune_layers resnet_fc linear \
  --film \
  --add_in_film \
  --replace_BN all;
done
done
