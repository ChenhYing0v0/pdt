set -e

# 变量定义
DATA_ROOT=./dataset
OUTPUT_DIR=./exp_results/OLinear

model_name=OLinear
seed=2023
dataset=weather

JOB_DIR=$OUTPUT_DIR/${model_name}_${dataset}
mkdir -p $JOB_DIR

LOG_DIR=$JOB_DIR/log
mkdir -p $LOG_DIR

CHECKPOINTS=$JOB_DIR/checkpoints/
RESULTS=$JOB_DIR/results/
TEST_RESULTS=$JOB_DIR/test_results/
LOG_PATH=$JOB_DIR/result_long_term_forecast.txt

# 预测长度与d_model映射
# 需要bash 4.0及以上

# declare -A d_model_map
# pred_len : d_model
# 可根据需要扩展
# 例如：d_model_map[48]=128
# d_model_map=(
#     [96]=512
#     [192]=512
#     [336]=1024
#     [720]=2048
# )

# 所有预测长度列表
pl_list=(96 192 336 720)


num_exp=${#pl_list[@]}

for ((i=0; i<$num_exp; i++)); do
    pl=${pl_list[$i]}

    echo "Start experiment: pred_len=$pl"
    CUDA_VISIBLE_DEVICES=0 python -u run_IN.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path $DATA_ROOT/weather/ \
        --data_path weather.csv \
        --q_mat_file ./dataset/cov_mats/weather/weather_96_ratio0.7.npy \
        --q_out_mat_file ./dataset/PCCA_mats/weather/weather_PCCA_OUT_H${pl}_avg.npy \
        --Q_chan_indep 0 \
        --model_id "${dataset}_96_${pl}" \
        --model ${model_name} \
        --data custom \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len ${pl} \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --des 'Exp' \
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
        --learning_rate 1e-3 \
        --lradj type3 \
        --loss_mode L1 \
        --patience 5 \
        --train_epochs 30 \
        --CKA_flag 0 \
        --dropout 0.0 \
        | sed -r "s/\\x1B\\[[0-9;]*[mGKHF]//g" >> $LOG_DIR/pred_len_${pl}.log 2>&1

    echo "Finished experiment: pred_len=$pl"
done

echo "All finished ..."

# --q_out_mat_file ./dataset/cov_mats/weather/weather_${pl}_ratio0.7.npy \