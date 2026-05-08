set -e

# 变量定义
DATA_ROOT=./dataset
OUTPUT_DIR=./exp_results/ltf_overall

model_name=DLinear
seed=2023
dataset=ETTh1
batch_size=32


JOB_DIR=$OUTPUT_DIR/${model_name}_${dataset}
mkdir -p $JOB_DIR

LOG_DIR=$JOB_DIR/log
mkdir -p $LOG_DIR

CHECKPOINTS=$JOB_DIR/checkpoints/
RESULTS=$JOB_DIR/results/
TEST_RESULTS=$JOB_DIR/test_results/
LOG_PATH=$JOB_DIR/result_long_term_forecast.txt


pl_list_gpu0=(96 192)
pl_list_gpu1=(336 720)

learning_rate_gpu0=(5e-4 5e-4)
learning_rate_gpu1=(5e-4 5e-4)

num_exp=${#pl_list_gpu0[@]}

for ((i=0; i<$num_exp; i++)); do
    pl0=${pl_list_gpu0[$i]}
    pl1=${pl_list_gpu1[$i]}

    lr0=${learning_rate_gpu0[$i]}
    lr1=${learning_rate_gpu1[$i]}


    echo "Start experiment: pred_len=$pl0 on GPU 0"
    CUDA_VISIBLE_DEVICES=0 python -u run_IN.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path $DATA_ROOT/ETT-small/ \
        --data_path ETTh1.csv \
        --checkpoints $CHECKPOINTS \
        --results $RESULTS \
        --test_results $TEST_RESULTS \
        --log_path $LOG_PATH \
        --model_id "${dataset}_96_${pl0}" \
        --model ${model_name} \
        --data ETTh1 \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len ${pl0} \
        --e_layers 2 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --batch_size ${batch_size} \
        --itr 1 \
        --fix_seed ${seed} \
        --learning_rate ${lr0} \
        --lradj type1 \
        --loss_mode L1 \
        --patience 8 \
        --train_epochs 50 \
        --dropout 0.2 \
        | sed -r "s/\\x1B\\[[0-9;]*[mGKHF]//g" >> $LOG_DIR/gpu_0.log 2>&1 &
    pid0=$!

    echo "Start experiment: pred_len=$pl1 on GPU 1"
    CUDA_VISIBLE_DEVICES=1 python -u run_IN.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path $DATA_ROOT/ETT-small/ \
        --data_path ETTh1.csv \
        --checkpoints $CHECKPOINTS \
        --results $RESULTS \
        --test_results $TEST_RESULTS \
        --log_path $LOG_PATH \
        --model_id "${dataset}_96_${pl1}" \
        --model ${model_name} \
        --data ETTh1 \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len ${pl1} \
        --e_layers 2 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --batch_size ${batch_size} \
        --itr 1 \
        --fix_seed ${seed} \
        --learning_rate ${lr1} \
        --lradj type1 \
        --loss_mode L1 \
        --patience 8 \
        --train_epochs 50 \
        --dropout 0.2 \
        | sed -r "s/\\x1B\\[[0-9;]*[mGKHF]//g" >> $LOG_DIR/gpu_1.log 2>&1 &
    pid1=$!

    # 等待本轮两个实验都结束
    wait $pid0
    wait $pid1
    echo "Finished experiments: pred_len=$pl0 on GPU 0, pred_len=$pl1 on GPU 1"
done

echo "All finished ..."

