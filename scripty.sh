#!/bin/sh

tgt_batch_size=16
n_target_samples=(20 50 100 200 500 1000 2000 5000)
train_seeds=(8 16 31)

for i in ${!n_target_samples[@]}; do
for value in {0..2}; do
  python ./train_classifier.py \
  --n_target_samples ${n_target_samples[$i]} \
  --target_batch_size $tgt_batch_size \
  --exp_dir fixed_batch_size_drop_last/baseline_s_20k_t_${n_target_samples[$i]} \
  --iter_idx $value \
  --train_seed ${train_seeds[$value]} \
  --load_trained_model \
  --model_path experiments/baseline/fixed_batch_size/baseline_s_20k_t_20/source_checkpoint_$value.pt;
  python ./train_classifier.py \
  --n_target_samples ${n_target_samples[$i]} \
  --target_batch_size $tgt_batch_size \
  --exp_dir fixed_batch_size_drop_last/fine_tune_2topmost_s_20k_t_${n_target_samples[$i]} \
  --iter_idx $value \
  --train_seed ${train_seeds[$value]} \
  --load_trained_model \
  --model_path experiments/baseline/fixed_batch_size/baseline_s_20k_t_20/source_checkpoint_$value.pt \
  --freeze \
  --fine_tune_layers resnet_fc linear;
  python ./train_classifier.py \
  --n_target_samples ${n_target_samples[$i]} \
  --target_batch_size $tgt_batch_size \
  --exp_dir fixed_batch_size_drop_last/fine_tune_linear_s_20k_t_${n_target_samples[$i]} \
  --iter_idx $value \
  --train_seed ${train_seeds[$value]} \
  --load_trained_model \
  --model_path experiments/baseline/fixed_batch_size/baseline_s_20k_t_20/source_checkpoint_$value.pt \
  --freeze \
  --fine_tune_layers linear;
done
done
