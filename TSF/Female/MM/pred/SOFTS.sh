#!/bin/bash
mkdir -p ./logs/MM/Female/seq/SOFTS
log_dir="./logs/MM/Female/seq/SOFTS/"

export CUDA_VISIBLE_DEVICES=1
model_name=SOFTS
seq_lens=(400)
pred_lens=(100 200 300 400)
bss=(128)
lrs=(1e-3)
layers=(2)
dropouts=(0.)
d_models=(256)

for bs in "${bss[@]}"; do
    for lr in "${lrs[@]}"; do
        for layer in "${layers[@]}"; do
            for dropout in "${dropouts[@]}"; do
                for d_model in "${d_models[@]}"; do
                    for pred_len in "${pred_lens[@]}"; do
                        for seq_len in "${seq_lens[@]}"; do
                                python -u run.py \
                                --data_path ./dataset/log_Female_Mortality.csv \
                                --data Mortality \
                                --task MM\
                                --model $model_name \
                                --seq_len $seq_len \
                                --pred_len $pred_len \
                                --batch_size $bs \
                                --learning_rate $lr \
                                --layers $layer\
                                --dropout $dropout\
                                --d_model $d_model\
                                --use_norm 1 >"${log_dir}seq_${seq_len}_pred_${pred_len}.log"
                        done
                    done
                done
            done
        done
    done
done
