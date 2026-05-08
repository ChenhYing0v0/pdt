set -e

# 变量定义
DATA_ROOT=./dataset
OUTPUT_DIR=./exp_results/PLinear_D

model_name=PLinear_D
seed=2023
dataset=traffic

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

k_dim_list=(96 96 96 96)
# k_dim_list=(8)

film_hidden_list=(48 48 48 48)

num_exp=${#pl_list[@]}

for ((i=0; i<$num_exp; i++)); do
    pl=${pl_list[$i]}
    k_dim=${k_dim_list[$i]}
    film_hidden=${film_hidden_list[$i]}
    echo "Start experiment: pred_len=$pl"
    CUDA_VISIBLE_DEVICES=0 python -u run_IN.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path $DATA_ROOT/traffic/ \
        --data_path traffic.csv \
        --q_mat_file ./dataset/PCCA_mats/traffic/traffic_PCCA_L96_K${k_dim}_H${pl}_avg.npy \
        --q_out_mat_file ./dataset/PCCA_mats/traffic/traffic_PCCA_OUT_L96_K${k_dim}_H${pl}_svdB_avg.npy \
        --Q_MAT_file ./dataset/PCCA_mats/traffic/traffic_PCCA_L96_K${k_dim}_H${pl}_perchan.npy \
        --Q_OUT_MAT_file ./dataset/PCCA_mats/traffic/traffic_PCCA_OUT_H${pl}_perchan.npy \
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
<<<<<<< HEAD
        --des 'Exp_CAR' \
        --embed_size 16 \
        --d_model 1024 \
        --d_ff 1024 \
=======
        --des 'Exp_CAR_twodrop_512d' \
        --embed_size 16 \
        --d_model 512 \
        --d_ff 512 \
>>>>>>> 4ddd9c7d1d205df2a3df37e4e559904fd0692119
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
        --k_dim ${k_dim} \
        --gate_module CAR \
        --car_reduction 16 \
        --pla_film_hidden ${film_hidden} \
        --pla_eps 0.2 \
        --pla_dropout 0.0 \
        | sed -r "s/\\x1B\\[[0-9;]*[mGKHF]//g" >> $LOG_DIR/pred_len_${pl}.log 2>&1

    echo "Finished experiment: pred_len=$pl"
done

echo "All finished ..."

