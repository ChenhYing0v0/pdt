set -e

# 变量定义
DATA_ROOT=./dataset
OUTPUT_DIR=./exp_results/PCCA_OLinear

model_name=PCCA_OLinear
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
# pl_list=(720)

# 对应的学习率
learning_rate_list=(5e-4 5e-4 5e-4 1e-4)
# learning_rate_list=(1e-4)

num_exp=${#pl_list[@]}

k_dim_list=(96 96 96 96)
# k_dim_list=(8)

# film_hidden_list=(48 48 48 6)


for ((i=0; i<$num_exp; i++)); do
    pl=${pl_list[$i]}
    lr=${learning_rate_list[$i]}
    k_dim=${k_dim_list[$i]}
    echo "Start experiment: pred_len=$pl"
    CUDA_VISIBLE_DEVICES=2 python -u run_IN.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path $DATA_ROOT/ETT-small/ \
        --data_path ETTh1.csv \
        --q_mat_file ./dataset/PCCA_mats/ETTh1/ETTh1_PCCA_L96_K${k_dim}_H${pl}_avg_new.npy \
        --q_out_mat_file ./dataset/PCCA_mats/ETTh1/ETTh1_PCCA_OUT_L96_K${k_dim}_H${pl}_svdB_avg_new.npy \
        --Q_MAT_file ./dataset/PCCA_mats/ETTh1/ETTh1_PCCA_L96_K${k_dim}_H${pl}_perchan.npy \
        --Q_OUT_MAT_file ./dataset/PCCA_mats/ETTh1/ETTh1_PCCA_OUT_H${pl}_perchan.npy \
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
        --des 'Exp_QinBefore_newQout2.0_newQin' \
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
        --k_dim ${k_dim} \
        --gate_module None \
        | sed -r "s/\\x1B\\[[0-9;]*[mGKHF]//g" >> $LOG_DIR/pred_len_${pl}.log 2>&1

    echo "Finished experiment: pred_len=$pl"
done

echo "All finished ..."

