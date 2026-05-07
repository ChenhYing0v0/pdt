#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASELINE_DIR=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
DATA_ROOT=${DATA_ROOT:-"$BASELINE_DIR/dataset"}
OUTPUT_DIR="$BASELINE_DIR/exp_results/PDT"

MODEL_NAME=PDT
SEED=2023
DATASET=weather
PL_LIST=(96 192 336 720)
R_RANK_LIST=(96 96 96 96)
K_TOP_LIST=(16 16 16 16)

JOB_DIR="$OUTPUT_DIR/${MODEL_NAME}_${DATASET}"
mkdir -p "$JOB_DIR/log"

for ((i=0; i<${#PL_LIST[@]}; i++)); do
  PL=${PL_LIST[$i]}
  R_RANK=${R_RANK_LIST[$i]}
  K_TOP=${K_TOP_LIST[$i]}

  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -u "$BASELINE_DIR/run.py" \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id "${DATASET}_96_${PL}" \
    --model "$MODEL_NAME" \
    --data custom \
    --root_path "$DATA_ROOT/weather/" \
    --data_path weather.csv \
    --features M \
    --target OT \
    --freq h \
    --seq_len 96 \
    --label_len 48 \
    --pred_len "$PL" \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --e_layers 2 \
    --d_layers 1 \
    --d_model 512 \
    --d_ff 512 \
    --factor 3 \
    --batch_size 32 \
    --itr 1 \
    --train_epochs 30 \
    --patience 5 \
    --learning_rate 1e-3 \
    --lradj type3 \
    --dropout 0.0 \
    --loss_mode L1 \
    --rec_lambda 1 \
    --auxi_lambda 0 \
    --seed "$SEED" \
    --des Exp_0.1a_Rk_Dnorm_k-EU-mask \
    --q_mat_file "dataset/RRR_mats/weather/weather_RRR_L96_R${R_RANK}_H${PL}_Qin.npy" \
    --q_out_mat_file "dataset/PCCA_mats/weather/weather_PCCA_OUT_H${PL}_identity_avg.npy" \
    --r_mat_file "dataset/RRR_mats/weather/weather_RRR_L96_R${R_RANK}_H${PL}_R.npy" \
    --rk_mat_file "dataset/RRR_mats/weather/weather_RRR_L96_R${K_TOP}_H${PL}_R.npy" \
    --r_rank "$R_RANK" \
    --k_top "$K_TOP" \
    --embed_size 16 \
    --alpha_init 0.1 \
    --output_dir "$JOB_DIR/pl_${PL}" \
    >> "$JOB_DIR/log/pred_len_${PL}.log" 2>&1
done
