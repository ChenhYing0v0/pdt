#!/bin/bash
set -e
set -o pipefail

# 超参数：mask_threshold 搜索空间
mask_thr_search_space=(0.05 0.1 0.2 0.3 0.4 0.5 0.7)

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

# 变量定义
DATA_ROOT=./dataset
OUTPUT_DIR=./exp_results/R2Linear_tuning_thr

model_name=R2Linear
seed=2023
dataset=ETTh1

JOB_DIR=$OUTPUT_DIR/${model_name}_${dataset}
mkdir -p $JOB_DIR

LOG_DIR=$JOB_DIR/log
mkdir -p $LOG_DIR

CHECKPOINTS=$JOB_DIR/checkpoints/
RESULTS=$JOB_DIR/results/
TEST_RESULTS=$JOB_DIR/test_results/
LOG_PATH=$JOB_DIR/result_long_term_forecast.txt

# 所有预测长度列表
pl_list=(96 192 336 720)

# 对应的学习率
learning_rate_list=(5e-4 5e-4 5e-4 1e-4)

# r_rank 固定为与示例一致
r_rank_list=(96 96 96 96)

# 其余与原脚本保持一致：固定 k_top 与 alpha_init 为常用默认值（与 R2Linear_ETTh1.sh 中一致）
# 如需更改，可在运行前通过环境变量覆盖：K_TOP, ALPHA_INIT
DEFAULT_K_TOP=16
DEFAULT_ALPHA_INIT=0.1
K_TOP=${K_TOP:-$DEFAULT_K_TOP}
ALPHA_INIT=${ALPHA_INIT:-$DEFAULT_ALPHA_INIT}

num_exp=${#pl_list[@]}

echo "=========================================================="
echo "Starting Threshold Tuning (mask_threshold) for ${model_name} on ${dataset}"
echo "Start time: $(date)"
echo "=========================================================="

for ((i=0; i<$num_exp; i++)); do
    pl=${pl_list[$i]}
    lr=${learning_rate_list[$i]}
    r_rank=${r_rank_list[$i]}

    # 为每个 pred_len 初始化最优指标和参数
    best_mse=999999.0
    best_mae=999999.0
    best_thr=0.0

    echo ""
    echo "----------------------------------------------------------"
    echo ">>>>> Starting Tuning for Prediction Length (pred_len) = ${pl}"
    echo "----------------------------------------------------------"

    # 遍历 mask_threshold
    for thr in "${mask_thr_search_space[@]}"; do
        echo ""
        echo "--- [RUNNING] pred_len=${pl}, mask_threshold=${thr}, k_top=${K_TOP}, alpha_init=${ALPHA_INIT} ---"

        # 动态生成本次运行的日志文件路径
        current_log_file="$LOG_DIR/pl_${pl}_thr_${thr}.log"

        # 执行Python脚本，并将输出重定向到日志文件
        python -u run_IN.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTh1.csv \
            --q_mat_file ./dataset/RRR_mats/ETTh1/ETTh1_RRR_L96_R${r_rank}_H${pl}_Qin.npy \
            --q_out_mat_file ./dataset/PCCA_mats/ETTh1/ETTh1_PCCA_OUT_H${pl}_identity_avg.npy \
            --r_mat_file ./dataset/RRR_mats/ETTh1/ETTh1_RRR_L96_R${r_rank}_H${pl}_R.npy \
            --rk_mat_file ./dataset/RRR_mats/ETTh1/ETTh1_RRR_L96_R${K_TOP}_H${pl}_R.npy \
            --Q_chan_indep 0 \
            --model_id "${dataset}_96_${pl}" \
            --model ${model_name} \
            --data ETTh1 \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --des "Tune_thr${thr}" \
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
            --lradj type1 \
            --loss_mode L1 \
            --patience 8 \
            --train_epochs 30 \
            --CKA_flag 0 \
            --dropout 0.2 \
            --freeze_R 0 \
            --r_rank ${r_rank} \
            --k_top ${K_TOP} \
            --alpha_init ${ALPHA_INIT} \
            --mask_threshold ${thr} \
            | sed -r "s/\\x1B\\[[0-9;]*[mGKHF]//g" > "$current_log_file" 2>&1

        # 从集中结果文件中提取本次 run 的 MSE 和 MAE（Python 会写两行：标识行+下一行的指标）
        run_marker_prefix="long_term_forecast_${dataset}_96_${pl}_"
        run_marker_suffix="Tune_thr${thr}"

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

        # 比较并更新最优结果（以 MSE 为主）
        if (( $(echo "$current_mse < $best_mse" | bc -l) )); then
            best_mse=$current_mse
            best_mae=$current_mae
            best_thr=$thr
            # 彩色输出（若可用）
            if [ -n "$COLOR_GREEN" ]; then
                printf "  %b[NEW BEST] Found new best threshold for pred_len=%s!%b\n" "$COLOR_GREEN" "$pl" "$COLOR_RESET"
                printf "  %b           mask_threshold=%s, MSE=%s, MAE=%s%b\n" \
                    "$COLOR_GREEN" "$best_thr" "$best_mse" "$best_mae" "$COLOR_RESET"
            else
                echo "  [NEW BEST] Found new best threshold for pred_len=${pl}!"
                echo "             mask_threshold=${best_thr}, MSE=${best_mse}, MAE=${best_mae}"
            fi
        fi
    done # mask_threshold 循环结束

    echo ""
    echo "=========================================================="
    echo ">>>>> Tuning Summary for pred_len = ${pl} <<<<<"
    echo "  Optimal mask_threshold: ${best_thr}"
    echo "  Best MSE: ${best_mse}"
    echo "  Best MAE: ${best_mae}"
    echo "=========================================================="

    # 将最优结果记录到主日志文件
    echo "pred_len=${pl}, mask_threshold=${best_thr}, mse=${best_mse}, mae=${best_mae}" >> "$LOG_PATH"

done

echo ""
echo "All threshold tuning experiments finished."
echo "End time: $(date)"
