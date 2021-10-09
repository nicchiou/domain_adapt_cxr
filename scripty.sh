#!/bin/sh

src_batch_size=16
tgt_batch_size=16
n_target_samples=(20 50 100 200 500 1000 2000 5000)
train_seeds=(8 16 31)

for i in ${!n_target_samples[@]}; do
for value in {0..2}; do
  python ./train_classifier.py \
  --early_stop \
  --n_target_samples ${n_target_samples[$i]} \
  --source_batch_size $src_batch_size \
  --target_batch_size $tgt_batch_size \
  --exp_dir baseline/fixed_batch_size/baseline_s_20k_t_${n_target_samples[$i]} \
  --load_trained_model \
  --model_path experiments/baseline/fixed_batch_size/baseline_s_20k_t_20/source_checkpoint_$value.pt \
  --iter_idx $value \
  --train_seed ${train_seeds[$value]};
done
done