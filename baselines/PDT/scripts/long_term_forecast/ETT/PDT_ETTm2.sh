#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASELINE_DIR=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
DATA_ROOT=${DATA_ROOT:-"$BASELINE_DIR/dataset"}
OUTPUT_DIR="$BASELINE_DIR/exp_results/PDT"

MODEL_NAME=PDT
SEED=2023
DATASET=ETTm2
PL_LIST=(96 192 336 720)
LEARNING_RATE_LIST=(1e-4 1e-4 1e-4 1e-4)
DROPOUT_LIST=(0.1 0.1 0.1 0.2)
R_RANK_LIST=(48 48 48 48)
K_TOP_LIST=(48 8 32 32)
ALPHA_INIT_LIST=(0.7 0.5 0.35 0.5)
MASK_THRESHOLD_LIST=(0.15 0.01 0.01 0.05)

JOB_DIR="$OUTPUT_DIR/${MODEL_NAME}_${DATASET}"
mkdir -p "$JOB_DIR/log"

for ((i=0; i<${#PL_LIST[@]}; i++)); do
  PL=${PL_LIST[$i]}
  LR=${LEARNING_RATE_LIST[$i]}
  DROPOUT=${DROPOUT_LIST[$i]}
  R_RANK=${R_RANK_LIST[$i]}
  K_TOP=${K_TOP_LIST[$i]}
  ALPHA=${ALPHA_INIT_LIST[$i]}
  THR=${MASK_THRESHOLD_LIST[$i]}

  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -u "$BASELINE_DIR/run.py" \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id "${DATASET}_48_${PL}" \
    --model "$MODEL_NAME" \
    --data ETTm2 \
    --root_path "$DATA_ROOT/ETT-small/" \
    --data_path ETTm2.csv \
    --features M \
    --target OT \
    --freq min \
    --seq_len 48 \
    --label_len 48 \
    --pred_len "$PL" \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --e_layers 2 \
    --d_layers 1 \
    --d_model 512 \
    --d_ff 512 \
    --factor 3 \
    --batch_size 32 \
    --itr 1 \
    --train_epochs 30 \
    --patience 8 \
    --learning_rate "$LR" \
    --lradj type1 \
    --dropout "$DROPOUT" \
    --loss_mode L1 \
    --rec_lambda 1 \
    --auxi_lambda 0 \
    --seed "$SEED" \
    --des Exp_final_48 \
    --q_mat_file "dataset/RRR_mats/ETTm2/ETTm2_RRR_L48_R${R_RANK}_H${PL}_Qin.npy" \
    --q_out_mat_file "dataset/PCCA_mats/ETTm2/ETTm2_PCCA_OUT_H${PL}_identity_avg.npy" \
    --r_mat_file "dataset/RRR_mats/ETTm2/ETTm2_RRR_L48_R${R_RANK}_H${PL}_R.npy" \
    --rk_mat_file "dataset/RRR_mats/ETTm2/ETTm2_RRR_L48_R${K_TOP}_H${PL}_R.npy" \
    --r_rank "$R_RANK" \
    --k_top "$K_TOP" \
    --embed_size 16 \
    --alpha_init "$ALPHA" \
    --mask_threshold "$THR" \
    --output_dir "$JOB_DIR/pl_${PL}" \
    >> "$JOB_DIR/log/pred_len_${PL}.log" 2>&1
done
