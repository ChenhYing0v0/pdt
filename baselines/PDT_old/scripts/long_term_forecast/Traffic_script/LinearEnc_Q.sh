set -e

# 变量定义
DATA_ROOT=./dataset
OUTPUT_DIR=./exp_results/LinearEnc_Q

model_name=LinearEnc_Q
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

# 预测长度列表
pl_list=(96 192 336 720)

num_exp=${#pl_list[@]}

for ((i=0; i<$num_exp; i++)); do
    pl=${pl_list[$i]}

    echo "Start experiment: pred_len=$pl"
    CUDA_VISIBLE_DEVICES=0 python -u run_IN.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path $DATA_ROOT/traffic/ \
        --data_path traffic.csv \
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
        --des 'Exp_cat2_ln' \
        --embed_size 1 \
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
        --concat 2 \
        --T 5 \
        --eps 1e-5 \
        --momentum 0.1 \
        --affine 0 \
        | sed -r "s/\\x1B\\[[0-9;]*[mGKHF]//g" >> $LOG_DIR/pred_len_${pl}.log 2>&1

    echo "Finished experiment: pred_len=$pl"
done

echo "All finished ..." 