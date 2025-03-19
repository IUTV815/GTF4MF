#!/bin/bash
mkdir -p ./logs/SS/Female/seq/Leddam
log_dir="./logs/SS/Female/seq/Leddam/"

export CUDA_VISIBLE_DEVICES=1
model_name=Leddam
seq_lens=(100 200 300 400)
pred_lens=(400 1)
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
                                --task SS\
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
