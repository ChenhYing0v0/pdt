#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASELINE_DIR=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
DATA_ROOT=${DATA_ROOT:-"$BASELINE_DIR/dataset"}
OUTPUT_DIR="$BASELINE_DIR/exp_results/PDT"

MODEL_NAME=PDT
SEED=2023
DATASET=ETTh1
DATA_ARG=ETTh1
DATA_FILE=ETTh1.csv
DATA_SUBDIR=ETT-small
MATRIX_KEY=ETTh1
ENC_IN=7
TARGET=OT
FREQ=h
LADJ=type1
PL_LIST=(96 192 336 720)
LEARNING_RATE_LIST=(0.0005 0.0005 0.0005 0.0001)
D_MODEL_LIST=(512 512 512 512)
D_FF_LIST=(512 512 512 512)
E_LAYER_LIST=(2 2 2 2)
BATCH_SIZE_LIST=(32 32 32 32)
DROPOUT_LIST=(0.2 0.2 0.2 0.2)
R_RANK_LIST=(96 96 96 96)
K_TOP_LIST=(16 96 64 96)
ALPHA_INIT_LIST=(0.15 0.05 0.05 0.5)
MASK_THRESHOLD_LIST=(0.05 0.05 0.05 0.05)

JOB_DIR="$OUTPUT_DIR/${MODEL_NAME}_${DATASET}"
mkdir -p "$JOB_DIR/log"

for ((i=0; i<${#PL_LIST[@]}; i++)); do
  PL=${PL_LIST[$i]}
  LR=${LEARNING_RATE_LIST[$i]}
  D_MODEL=${D_MODEL_LIST[$i]}
  D_FF=${D_FF_LIST[$i]}
  E_LAYERS=${E_LAYER_LIST[$i]}
  BATCH_SIZE=${BATCH_SIZE_LIST[$i]}
  DROPOUT=${DROPOUT_LIST[$i]}
  R_RANK=${R_RANK_LIST[$i]}
  K_TOP=${K_TOP_LIST[$i]}
  ALPHA=${ALPHA_INIT_LIST[$i]}
  THR=${MASK_THRESHOLD_LIST[$i]}

  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -u "$BASELINE_DIR/run.py" \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id "${DATASET}_96_${PL}" \
    --model "$MODEL_NAME" \
    --data "$DATA_ARG" \
    --root_path "$DATA_ROOT/${DATA_SUBDIR}/" \
    --data_path "$DATA_FILE" \
    --features M \
    --target "$TARGET" \
    --freq "$FREQ" \
    --seq_len 96 \
    --label_len 48 \
    --pred_len "$PL" \
    --enc_in "$ENC_IN" \
    --dec_in "$ENC_IN" \
    --c_out "$ENC_IN" \
    --e_layers "$E_LAYERS" \
    --d_layers 1 \
    --d_model "$D_MODEL" \
    --d_ff "$D_FF" \
    --factor 3 \
    --batch_size "$BATCH_SIZE" \
    --num_workers 8 \
    --itr 1 \
    --train_epochs 30 \
    --patience 8 \
    --learning_rate "$LR" \
    --lradj "$LADJ" \
    --dropout "$DROPOUT" \
    --loss_mode L1 \
    --rec_lambda 1 \
    --auxi_lambda 0 \
    --Q_chan_indep 0 \
    --freeze_R 0 \
    --CKA_flag 0 \
    --seed "$SEED" \
    --des FixedPDT \
    --q_mat_file "dataset/RRR_mats/${MATRIX_KEY}/${MATRIX_KEY}_RRR_L96_R${R_RANK}_H${PL}_Qin.npy" \
    --q_out_mat_file "dataset/PCCA_mats/${MATRIX_KEY}/${MATRIX_KEY}_PCCA_OUT_H${PL}_identity_avg.npy" \
    --r_mat_file "dataset/RRR_mats/${MATRIX_KEY}/${MATRIX_KEY}_RRR_L96_R${R_RANK}_H${PL}_R.npy" \
    --rk_mat_file "dataset/RRR_mats/${MATRIX_KEY}/${MATRIX_KEY}_RRR_L96_R${K_TOP}_H${PL}_R.npy" \
    --r_rank "$R_RANK" \
    --k_top "$K_TOP" \
    --embed_size 16 \
    --alpha_init "$ALPHA" \
    --mask_threshold "$THR" \
    --output_dir "$JOB_DIR/pl_${PL}" \
    >> "$JOB_DIR/log/pred_len_${PL}.log" 2>&1
done
