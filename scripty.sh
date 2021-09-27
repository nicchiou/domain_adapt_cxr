#!/bin/sh

tgt_batch_size=(4 8 8 8 16 16)
n_target_samples=(20 50 100 200 500 1000)
train_seeds=(8 16 31)

for i in ${!n_target_samples[@]}; do
for value in {0..2}; do
  python ./train_classifier.py \
  --n_target_samples ${n_target_samples[$i]} \
  --target_batch_size ${tgt_batch_size[$i]} \
  --exp_dir film_s_20k_t_${n_target_samples[$i]} \
  --iter_idx $value \
  --train_seed ${train_seeds[$value]} \
  --freeze \
  --fine_tune_layers resnet_fc linear \
  --film;
done
done
