#!/bin/sh

tgt_batch_size=16
n_target_samples=20
train_seeds=(8 16 31)

for value in {0..2}; do
  python ./train_classifier.py \
  --n_target_samples $n_target_samples \
  --target_batch_size $tgt_batch_size \
  --exp_dir film_s_20k_t_$n_target_samples \
  --iter_idx $value \
  --train_seed ${train_seeds[$value]} \
  --freeze \
  --fine_tune_layers resnet_fc linear \
  --film;
done
