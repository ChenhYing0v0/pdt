#!/bin/bash
set -e
set -o pipefail

# ==================== SEARCH SPACE ====================
# 超参数搜索空间（可按需调整）
# k_top_search_space=(8 16 32 64 96)
k_top_search_space=(96)

alpha_search_space=(0.01 0.05 0.15)
mask_thr_search_space=(0.10)

# GPU 选择：可通过外部环境变量覆盖，默认使用第 0 号 GPU
GPU_DEVICES="0,1,2"
export CUDA_VISIBLE_DEVICES=$GPU_DEVICES

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
# ==================== END SEARCH SPACE ====================

# 变量定义
DATA_ROOT=./dataset
OUTPUT_DIR=./exp_results/R2Linear_tuning

model_name=R2Linear
seed=2023
dataset=traffic

JOB_DIR=$OUTPUT_DIR/${model_name}_${dataset}
mkdir -p "$JOB_DIR"

LOG_DIR=$JOB_DIR/log
mkdir -p "$LOG_DIR"

CHECKPOINTS=$JOB_DIR/checkpoints/
RESULTS=$JOB_DIR/results/
TEST_RESULTS=$JOB_DIR/test_results/
LOG_PATH=$JOB_DIR/result_long_term_forecast.txt

echo "=========================================================="
echo "Starting Hyperparameter Tuning for ${model_name} on ${dataset}"
echo "Using GPU(s): ${CUDA_VISIBLE_DEVICES}"
echo "Start time: $(date)"
echo "=========================================================="

# 预测长度列表（参考 traffic 基线）
pl_list=(96 192 336 720)
# pl_list=(720)

# 对应学习率、秩设置（沿用 traffic 基线设置）
learning_rate_list=(5e-4 5e-4 5e-4 5e-4)
# learning_rate_list=(5e-4)
r_rank_list=(96 96 96 96)
# r_rank_list=(96)

num_exp=${#pl_list[@]}

for ((i=0; i<${num_exp}; i++)); do
    pl=${pl_list[$i]}
    lr=${learning_rate_list[$i]}
    r_rank=${r_rank_list[$i]}

    # 为每个 pred_len 初始化最优指标和参数
    best_mse=999999.0
    best_mae=999999.0
    best_k_top=0
    best_alpha=0.0
    best_thr=0.0

    echo ""
    echo "----------------------------------------------------------"
    echo ">>>>> Starting Tuning for Prediction Length (pred_len) = ${pl}"
    echo "----------------------------------------------------------"

    for k_top in "${k_top_search_space[@]}"; do
        for alpha in "${alpha_search_space[@]}"; do
            for thr in "${mask_thr_search_space[@]}"; do
                echo ""
                echo "--- [RUNNING] pred_len=${pl}, k_top=${k_top}, alpha_init=${alpha}, mask_threshold=${thr} ---"

                current_log_file="$LOG_DIR/pl_${pl}_k_${k_top}_alpha_${alpha}_thr_${thr}.log"

                # 执行训练（去除 ANSI 颜色后重定向到本次运行日志文件）
                python -u run_IN.py \
                --task_name long_term_forecast \
                --is_training 1 \
                --root_path $DATA_ROOT/traffic/ \
                --data_path traffic.csv \
                --q_mat_file ./dataset/RRR_mats/traffic/traffic_RRR_L96_R${r_rank}_H${pl}_Qin.npy \
                --q_out_mat_file ./dataset/PCCA_mats/traffic/traffic_PCCA_OUT_H${pl}_identity_avg.npy \
                --r_mat_file ./dataset/RRR_mats/traffic/traffic_RRR_L96_R${r_rank}_H${pl}_R.npy \
                --rk_mat_file ./dataset/RRR_mats/traffic/traffic_RRR_L96_R${k_top}_H${pl}_R.npy \
                --Q_chan_indep 0 \
                --model_id "${dataset}_96_${pl}" \
                --model ${model_name} \
                --data custom \
                --features M \
                --seq_len 96 \
                --label_len 48 \
                --pred_len ${pl} \
                --e_layers 3 \
                --d_layers 1 \
                --factor 3 \
                --enc_in 862 \
                --dec_in 862 \
                --c_out 862 \
                --des "Tune_k${k_top}_a${alpha}_thr${thr}" \
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
                --mask_threshold ${thr} \
                --use_multi_gpu \
                --devices $GPU_DEVICES \
                | sed -r "s/\\x1B\\[[0-9;]*[mGKHF]//g" > "$current_log_file" 2>&1

                # 从集中结果文件中提取本次 run 的 MSE 和 MAE
                run_marker_prefix="long_term_forecast_${dataset}_96_${pl}_"
                run_marker_suffix="Tune_k${k_top}_a${alpha}_thr${thr}"

                metrics_line=""
                for attempt in 1 2 3 4 5 6 7 8 8 10; do
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
                    best_thr=$thr
                    if [ -n "$COLOR_GREEN" ]; then
                        printf "  %b[NEW BEST] Found new best parameters for pred_len=%s!%b\n" "$COLOR_GREEN" "$pl" "$COLOR_RESET"
                        printf "  %b           k_top=%s, alpha_init=%s, mask_threshold=%s, MSE=%s, MAE=%s%b\n" \
                            "$COLOR_GREEN" "$best_k_top" "$best_alpha" "$best_thr" "$best_mse" "$best_mae" "$COLOR_RESET"
                    else
                        echo "  [NEW BEST] Found new best parameters for pred_len=${pl}!"
                        echo "             k_top=${best_k_top}, alpha_init=${best_alpha}, mask_threshold=${best_thr}, MSE=${best_mse}, MAE=${best_mae}"
                    fi
                fi
            done
        done
    done

    echo ""
    echo "=========================================================="
    echo ">>>>> Tuning Summary for pred_len = ${pl} <<<<<"
    echo "  Optimal k_top: ${best_k_top}"
    echo "  Optimal alpha_init: ${best_alpha}"
    echo "  Optimal mask_threshold: ${best_thr}"
    echo "  Best MSE: ${best_mse}"
    echo "  Best MAE: ${best_mae}"
    echo "=========================================================="

    echo "pred_len=${pl}, k_top=${best_k_top}, alpha_init=${best_alpha}, mask_threshold=${best_thr}, mse=${best_mse}, mae=${best_mae}" >> "$LOG_PATH"

done

echo ""
echo "All tuning experiments finished."
echo "End time: $(date)"