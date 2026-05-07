#!/bin/bash
set -e
set -o pipefail

# ==================== MODIFICATION START ====================
# 定义超参数搜索空间
k_top_search_space=(8 16 32 64 96)
alpha_search_space=(0.05 0.15 0.25 0.35 0.5)
# mask_thr_search_space=(0.05)
mask_thr_search_space=(0.05 0.15 0.25 0.35)

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
# ===================== MODIFICATION END =====================

# 变量定义
DATA_ROOT=./dataset
OUTPUT_DIR=./exp_results/R2Linear_tuning

model_name=R2Linear
seed=2023
dataset=Exchange

JOB_DIR=$OUTPUT_DIR/${model_name}_${dataset}
mkdir -p $JOB_DIR

LOG_DIR=$JOB_DIR/log
mkdir -p $LOG_DIR

CHECKPOINTS=$JOB_DIR/checkpoints/
RESULTS=$JOB_DIR/results/
TEST_RESULTS=$JOB_DIR/test_results/
LOG_PATH=$JOB_DIR/result_long_term_forecast.txt

# 固定使用第 0 号 GPU
export CUDA_VISIBLE_DEVICES=0

# 所有预测长度列表
pl_list=(96 192 336 720)

# 对应的学习率
learning_rate_list=(1e-4 1e-4 1e-4 1e-4)
d_models_list=(256 128 256 256)

r_rank_list=(96 96 96 96)
# k_top_list is now replaced by the search space

num_exp=${#pl_list[@]}

echo "=========================================================="
echo "Starting Hyperparameter Tuning for ${model_name} on ${dataset}"
echo "Start time: $(date)"
echo "=========================================================="

for ((i=0; i<$num_exp; i++)); do
    pl=${pl_list[$i]}
    lr=${learning_rate_list[$i]}
    r_rank=${r_rank_list[$i]}
    d_model=${d_models_list[$i]}
    
    # ==================== MODIFICATION START ====================
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

    # 遍历 k_top
    for k_top in "${k_top_search_space[@]}"; do
        # 遍历 alpha_init
        for alpha in "${alpha_search_space[@]}"; do
            # 遍历 mask_threshold
            for thr in "${mask_thr_search_space[@]}"; do
            
            echo ""
            echo "--- [RUNNING] pred_len=${pl}, k_top=${k_top}, alpha_init=${alpha}, mask_threshold=${thr} ---"
            
            # 动态生成本次运行的日志文件路径
            current_log_file="$LOG_DIR/pl_${pl}_k_${k_top}_alpha_${alpha}_thr_${thr}.log"
            
            # 执行Python脚本，并将输出重定向到日志文件
            # 注意：我们将 k_top 和 alpha_init 作为变量传入
            python -u run_IN.py \
                --task_name long_term_forecast \
                --is_training 1 \
                --root_path $DATA_ROOT/exchange_rate/ \
                --data_path exchange_rate.csv \
                --q_mat_file ./dataset/RRR_mats/exchange_rate/exchange_rate_RRR_L96_R${r_rank}_H${pl}_Qin.npy \
                --q_out_mat_file ./dataset/PCCA_mats/exchange_rate/exchange_rate_PCCA_OUT_H${pl}_identity_avg.npy \
                --r_mat_file ./dataset/RRR_mats/exchange_rate/exchange_rate_RRR_L96_R${r_rank}_H${pl}_R.npy \
                --rk_mat_file ./dataset/RRR_mats/exchange_rate/exchange_rate_RRR_L96_R${k_top}_H${pl}_R.npy \
                --Q_chan_indep 0 \
                --model_id "${dataset}_96_${pl}" \
                --model ${model_name} \
                --data custom \
                --features M \
                --seq_len 96 \
                --label_len 48 \
                --pred_len ${pl} \
                --e_layers 2 \
                --factor 3 \
                --enc_in 8 \
                --dec_in 8 \
                --c_out 8 \
                --des "Tune_k${k_top}_a${alpha}_thr${thr}" \
                --embed_size 16 \
                --d_model ${d_model} \
                --d_ff ${d_model} \
                --batch_size 16 \
                --itr 1 \
                --auxi_lambda 0 \
                --rec_lambda 1 \
                --fix_seed ${seed} \
                --checkpoints $CHECKPOINTS \
                --results $RESULTS \
                --test_results $TEST_RESULTS \
                --log_path $LOG_PATH \
                --learning_rate ${lr} \
                --lradj type1 \
                --loss_mode L1 \
                --patience 8 \
                --train_epochs 1 \
                --CKA_flag 0 \
                --dropout 0.0 \
                --freeze_R 0 \
                --r_rank ${r_rank} \
                --k_top ${k_top} \
                --alpha_init ${alpha} \
                --mask_threshold ${thr} \
                | sed -r "s/\\x1B\\[[0-9;]*[mGKHF]//g" > "$current_log_file" 2>&1

            # 从集中结果文件中提取本次 run 的 MSE 和 MAE（Python 会写两行：标识行+下一行的指标）
            run_marker_prefix="long_term_forecast_${dataset}_96_${pl}_"
            run_marker_suffix="Tune_k${k_top}_a${alpha}_thr${thr}"

            metrics_line=""
            for attempt in 1 2 3 4 5 6 7 8 9 10; do
                # 找到包含本次 run 标识的最后一条记录的行号
                marker_line_no=$(grep -n "${run_marker_prefix}.*${run_marker_suffix}" "$LOG_PATH" | tail -n 1 | cut -d: -f1 || true)
                if [ -n "$marker_line_no" ]; then
                    # 指标位于标识行的下一行
                    metrics_line=$(sed -n "$((marker_line_no + 1))p" "$LOG_PATH" || true)
                fi
                # 兜底：如果未找到标识或下一行为空，尝试使用最近的包含 mse 的行
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

            # 解析数值
            current_mse=$(echo "$metrics_line" | sed -E 's/.*mse:([0-9.eE+-]+).*/\1/')
            current_mae=$(echo "$metrics_line" | sed -E 's/.*mae:([0-9.eE+-]+).*/\1/')

            # 数值校验
            if ! printf '%s' "$current_mse" | grep -Eq '^[+-]?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?$'; then
                echo "  [WARNING] Parsed MSE is not a number: '$current_mse' (line: $metrics_line)"
                continue
            fi
            if ! printf '%s' "$current_mae" | grep -Eq '^[+-]?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?$'; then
                echo "  [WARNING] Parsed MAE is not a number: '$current_mae' (line: $metrics_line)"
                continue
            fi

            printf "  [RESULT] MSE: %s, MAE: %s\n" "$current_mse" "$current_mae"

            # 比较并更新最优结果
            # 使用 `bc` 来处理浮点数比较
            if (( $(echo "$current_mse < $best_mse" | bc -l) )); then
                best_mse=$current_mse
                best_mae=$current_mae
                best_k_top=$k_top
                best_alpha=$alpha
                best_thr=$thr
                # 彩色输出（若可用）
                if [ -n "$COLOR_GREEN" ]; then
                    printf "  %b[NEW BEST] Found new best parameters for pred_len=%s!%b\n" "$COLOR_GREEN" "$pl" "$COLOR_RESET"
                    printf "  %b           k_top=%s, alpha_init=%s, mask_threshold=%s, MSE=%s, MAE=%s%b\n" \
                        "$COLOR_GREEN" "$best_k_top" "$best_alpha" "$best_thr" "$best_mse" "$best_mae" "$COLOR_RESET"
                else
                    echo "  [NEW BEST] Found new best parameters for pred_len=${pl}!"
                    echo "             k_top=${best_k_top}, alpha_init=${best_alpha}, mask_threshold=${best_thr}, MSE=${best_mse}, MAE=${best_mae}"
                fi
            fi
            done # mask_threshold 循环结束
        done # alpha_init 循环结束
    done # k_top 循环结束

    echo ""
    echo "=========================================================="
    echo ">>>>> Tuning Summary for pred_len = ${pl} <<<<<"
    echo "  Optimal k_top: ${best_k_top}"
    echo "  Optimal alpha_init: ${best_alpha}"
    echo "  Optimal mask_threshold: ${best_thr}"
    echo "  Best MSE: ${best_mse}"
    echo "  Best MAE: ${best_mae}"
    echo "=========================================================="
    
    # 将最优结果记录到主日志文件
    echo "pred_len=${pl}, k_top=${best_k_top}, alpha_init=${best_alpha}, mask_threshold=${best_thr}, mse=${best_mse}, mae=${best_mae}" >> "$LOG_PATH"

    # ===================== MODIFICATION END =====================
done

echo ""
echo "All tuning experiments finished."
echo "End time: $(date)"