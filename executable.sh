#!/bin/bash

cd /home/nschiou2/domain_adapt_cxr/
echo "Script executed from: ${PWD}"

src_batch_size=16
tgt_batch_size=16
n_target_samples=20

pipenv run python ./train_classifier.py \
  --early_stop \
  --n_target_samples $n_target_samples \
  --source_batch_size $src_batch_size \
  --target_batch_size $tgt_batch_size \
  --exp_dir film/fixed_batch_size/film_layer1_2topmost/film_s_20k_t_$n_target_samples \
  --iter_idx 2 \
  --train_seed 31 \
  --freeze \
  --fine_tune_layers resnet_fc linear \
  --film \
  --film_layers 1 2 3;