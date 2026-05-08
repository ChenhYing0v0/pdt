set -e

# 变量定义
DATA_ROOT=./dataset
OUTPUT_DIR=./exp_results/R2Linear

model_name=R2Linear
seed=2023
dataset=ECL

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
echo "${model_name} on ${dataset}"
echo "Start time: $(date)"

# 所有预测长度列表
# pl_list=(96 192 336 720)
pl_list=(96)

# r_rank_list=(96 96 96 96)
r_rank_list=(96)
# k_top_list=(16 16 16 16)
k_top_list=(96)

num_exp=${#pl_list[@]}

for ((i=0; i<$num_exp; i++)); do
    pl=${pl_list[$i]}
    r_rank=${r_rank_list[$i]}
    k_top=${k_top_list[$i]}
    echo "Start experiment: pred_len=$pl"
    CUDA_VISIBLE_DEVICES=2 python -u run_IN.py \
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
        --des 'Exp_0.1a_Rk_Dnorm_k-EU-mask' \
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
        --learning_rate 5e-4 \
        --lradj cosine\
        --loss_mode L1 \
        --patience 5 \
        --train_epochs 50 \
        --CKA_flag 0 \
        --dropout 0.0 \
        --freeze_R 0 \
        --r_rank ${r_rank} \
        --k_top ${k_top} \
        --alpha_init 0.05 \
        --mask_threshold 0.25 \
        | sed -r "s/\\x1B\\[[0-9;]*[mGKHF]//g" >> $LOG_DIR/pred_len_${pl}.log 2>&1

    echo "Finished experiment: pred_len=$pl"
done

echo "All finished ..."
echo "End time: $(date)"

# --q_out_mat_file ./dataset/cov_mats/ECL/electricity_${pl}_ratio0.7.npy \