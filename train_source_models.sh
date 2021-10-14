#!/bin/bash

src_batch_size=16
tgt_batch_size=16
n_target_samples=20
train_seeds=(8 16 31)

for value in {0..2}; do
  pipenv run python ./train_classifier.py \
  --early_stop \
  --n_target_samples $n_target_samples \
  --source_batch_size $src_batch_size \
  --target_batch_size $tgt_batch_size \
  --exp_dir baseline/fixed_batch_size/baseline_s_20k_t_$n_target_samples \
  --iter_idx $value \
  --train_seed ${train_seeds[$value]};
done
