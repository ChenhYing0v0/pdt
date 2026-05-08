#!/bin/bash
set -e
set -o pipefail

# ==================== HYPERPARAMETER SPACE ====================
# 可按需调整搜索范围
k_top_search_space=(8 16 24 32 48 64 96)
# k_top_search_space=(32 48 64 96)
alpha_search_space=(0.05 0.1 0.2 0.3 0.4 0.5 0.7)

# 调试开关（默认关闭）。若需要查看匹配到的指标行，可在运行前设置 DEBUG_LOG=1
DEBUG_LOG=${DEBUG_LOG:-0}

# 颜色输出（非 TTY 或重定向/nohup 时禁用颜色，避免日志出现 \033 乱码）
if [ -t 1 ] && [ -n "${TERM:-}" ] && command -v tput >/dev/null 2>&1; then
    if tput colors >/dev/null 2>&1; then
        COLOR_GREEN="$(tput setaf 2)"
        COLOR_BOLD="$(tput bold)"
        COLOR_RESET="$(tput sgr0)"
    else
        COLOR_GREEN=""; COLOR_BOLD=""; COLOR_RESET=""
    fi
else
    COLOR_GREEN=""; COLOR_BOLD=""; COLOR_RESET=""
fi
# ==============================================================

# =============== PATHS & CONSTANTS (ECL) ======================
DATA_ROOT=./dataset
OUTPUT_DIR=./exp_results/R2Linear_tuning

model_name=R2Linear
seed=2023
dataset=ECL

JOB_DIR=$OUTPUT_DIR/${model_name}_${dataset}
mkdir -p "$JOB_DIR"

LOG_DIR=$JOB_DIR/log
mkdir -p "$LOG_DIR"

CHECKPOINTS=$JOB_DIR/checkpoints/
RESULTS=$JOB_DIR/results/
TEST_RESULTS=$JOB_DIR/test_results/
LOG_PATH=$JOB_DIR/result_long_term_forecast.txt

# 固定使用第 0 号 GPU
export CUDA_VISIBLE_DEVICES=0

# 预测长度列表（与 baseline 脚本一致）
# pl_list=(96 192 336 720)
pl_list=(192 336 720)

# learning_rate_list=(5e-4 5e-4 5e-4 5e-4)
learning_rate_list=(5e-4 5e-4 5e-4)

# R 的秩（与 baseline 一致）
# r_rank_list=(96 96 96 96)
r_rank_list=(96 96 96)

echo "=========================================================="
echo "Starting Hyperparameter Tuning for ${model_name} on ${dataset}"
echo "Start time: $(date)"
echo "=========================================================="

num_exp=${#pl_list[@]}

for ((i=0; i<$num_exp; i++)); do
    pl=${pl_list[$i]}
    lr=${learning_rate_list[$i]}
    r_rank=${r_rank_list[$i]}

    best_mse=999999.0
    best_mae=999999.0
    best_k_top=0
    best_alpha=0.0

    echo ""
    echo "----------------------------------------------------------"
    echo ">>>>> Starting Tuning for Prediction Length (pred_len) = ${pl}"
    echo "----------------------------------------------------------"

    for k_top in "${k_top_search_space[@]}"; do
        for alpha in "${alpha_search_space[@]}"; do
            echo ""
            echo "--- [RUNNING] pred_len=${pl}, k_top=${k_top}, alpha_init=${alpha} ---"

            current_log_file="$LOG_DIR/pl_${pl}_k_${k_top}_alpha_${alpha}.log"

            # 运行训练（ECL 配置：data=custom, enc/dec/c_out=321）
            python -u run_IN.py \
                --task_name long_term_forecast \
                --is_training 1 \
                --root_path $DATA_ROOT/electricity/ \
                --data_path electricity.csv \
                --q_mat_file ./dataset/RRR_mats/electricity/electricity_RRR_L96_R${r_rank}_H${pl}_Qin.npy \
                --q_out_mat_file ./dataset/PCCA_mats/electricity/electricity_PCCA_OUT_H${pl}_identity_avg.npy \
                --r_mat_file ./dataset/RRR_mats/electricity/electricity_RRR_L96_R${r_rank}_H${pl}_R.npy \
                --rk_mat_file ./dataset/RRR_mats/electricity/electricity_RRR_L96_R${k_top}_H${pl}_R.npy \
                --Q_chan_indep 0 \
                --model_id "${dataset}_96_${pl}" \
                --model ${model_name} \
                --data custom \
                --features M \
                --seq_len 96 \
                --label_len 48 \
                --pred_len ${pl} \
                --e_layers 3 \
                --factor 3 \
                --enc_in 321 \
                --dec_in 321 \
                --c_out 321 \
                --des "Tune_k${k_top}_a${alpha}" \
                --embed_size 16 \
                --d_model 512 \
                --d_ff 512 \
                --batch_size 32 \
                --itr 1 \
                --auxi_lambda 0 \
                --rec_lambda 1 \
                --fix_seed ${seed} \
                --checkpoints $CHECKPOINTS \
                --results $RESULTS \
                --test_results $TEST_RESULTS \
                --log_path $LOG_PATH \
                --learning_rate ${lr} \
                --lradj cosine \
                --loss_mode L1 \
                --patience 5 \
                --train_epochs 50 \
                --CKA_flag 0 \
                --dropout 0.0 \
                --freeze_R 0 \
                --r_rank ${r_rank} \
                --k_top ${k_top} \
                --alpha_init ${alpha} \
                | sed -r "s/\\x1B\\[[0-9;]*[mGKHF]//g" > "$current_log_file" 2>&1

            # 提取指标：兼容 data=custom 的 run 标识（更宽松的匹配，避免固定 data 名称）
            run_marker_prefix="long_term_forecast_${dataset}_96_${pl}_"
            run_marker_suffix="Tune_k${k_top}_a${alpha}"

            metrics_line=""
            for attempt in 1 2 3 4 5 6 7 8 9 10; do
                marker_line_no=$(grep -n "${run_marker_prefix}.*${run_marker_suffix}" "$LOG_PATH" | tail -n 1 | cut -d: -f1 || true)
                if [ -n "$marker_line_no" ]; then
                    metrics_line=$(sed -n "$((marker_line_no + 1))p" "$LOG_PATH" || true)
                fi
                if [ -z "$metrics_line" ]; then
                    metrics_line=$(grep "mse:" "$LOG_PATH" | tail -n 1 || true)
                fi
                if [ -n "$metrics_line" ]; then
                    if [ "$DEBUG_LOG" = "1" ]; then
                        printf "  [DEBUG] Matched metrics line: %s\n" "$metrics_line"
                    fi
                    break
                fi
                sleep 1
            done

            if [ -z "$metrics_line" ]; then
                echo "  [WARNING] Could not extract metrics from results file. Check: ${LOG_PATH}"
                echo "            Current run log: ${current_log_file}"
                continue
            fi

            current_mse=$(echo "$metrics_line" | sed -E 's/.*mse:([0-9.eE+-]+).*/\1/')
            current_mae=$(echo "$metrics_line" | sed -E 's/.*mae:([0-9.eE+-]+).*/\1/')

            if ! printf '%s' "$current_mse" | grep -Eq '^[+-]?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?$'; then
                echo "  [WARNING] Parsed MSE is not a number: '$current_mse' (line: $metrics_line)"
                continue
            fi
            if ! printf '%s' "$current_mae" | grep -Eq '^[+-]?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?$'; then
                echo "  [WARNING] Parsed MAE is not a number: '$current_mae' (line: $metrics_line)"
                continue
            fi

            printf "  [RESULT] MSE: %s, MAE: %s\n" "$current_mse" "$current_mae"

            if (( $(echo "$current_mse < $best_mse" | bc -l) )); then
                best_mse=$current_mse
                best_mae=$current_mae
                best_k_top=$k_top
                best_alpha=$alpha
                if [ -n "$COLOR_GREEN" ]; then
                    printf "  %b[NEW BEST] Found new best parameters for pred_len=%s!%b\n" "$COLOR_GREEN" "$pl" "$COLOR_RESET"
                    printf "  %b           k_top=%s, alpha_init=%s, MSE=%s, MAE=%s%b\n" \
                        "$COLOR_GREEN" "$best_k_top" "$best_alpha" "$best_mse" "$best_mae" "$COLOR_RESET"
                else
                    echo "  [NEW BEST] Found new best parameters for pred_len=${pl}!"
                    echo "             k_top=${best_k_top}, alpha_init=${best_alpha}, MSE=${best_mse}, MAE=${best_mae}"
                fi
            fi
        done
    done

    echo ""
    echo "=========================================================="
    echo ">>>>> Tuning Summary for pred_len = ${pl} <<<<<"
    echo "  Optimal k_top: ${best_k_top}"
    echo "  Optimal alpha_init: ${best_alpha}"
    echo "  Best MSE: ${best_mse}"
    echo "  Best MAE: ${best_mae}"
    echo "=========================================================="

    echo "pred_len=${pl}, k_top=${best_k_top}, alpha_init=${best_alpha}, mse=${best_mse}, mae=${best_mae}" >> "$LOG_PATH"

done

echo ""
echo "All tuning experiments finished."
echo "End time: $(date)"
