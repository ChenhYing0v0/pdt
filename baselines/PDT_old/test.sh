set -e

# 变量定义
DATA_ROOT=./dataset
OUTPUT_DIR=./exp_results/iternormTS

model_name=DLinear
seed=2023
dataset=ETTh1

iter_norm=0
num_groups=1
T=9
revin=0
momentum=0.1
affine=0
add_module=None
individual=0

JOB_DIR=$OUTPUT_DIR/${model_name}_${dataset}_iternorm${iter_norm}_${add_module}_ind${individual}
mkdir -p $JOB_DIR

LOG_DIR=$JOB_DIR/log
mkdir -p $LOG_DIR

CHECKPOINTS=$JOB_DIR/checkpoints/
RESULTS=$JOB_DIR/results/
TEST_RESULTS=$JOB_DIR/test_results/
LOG_PATH=$JOB_DIR/result_long_term_forecast.txt

# 预测长度与d_model映射
# 需要bash 4.0及以上

declare -A d_model_map
# pred_len : d_model
# 可根据需要扩展
# 例如：d_model_map[48]=128
d_model_map=(
    [96]=512
    [192]=512
    [336]=1024
    [720]=2048
)

pl_list_gpu0=(96 192)
pl_list_gpu1=(336 720)

learning_rate_gpu0=(5e-4 5e-4)
learning_rate_gpu1=(5e-4 1e-4)

num_exp=${#pl_list_gpu0[@]}

for ((i=0; i<$num_exp; i++)); do
    pl0=${pl_list_gpu0[$i]}
    pl1=${pl_list_gpu1[$i]}

    lr0=${learning_rate_gpu0[$i]}
    lr1=${learning_rate_gpu1[$i]}

    d_model0=${d_model_map[$pl0]}
    d_model1=${d_model_map[$pl1]}

    echo "Start experiment: pred_len=$pl0 on GPU 0"
    CUDA_VISIBLE_DEVICES=0 python -u run_IN.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path $DATA_ROOT/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id "${dataset}_96_${pl0}" \
        --model ${model_name} \
        --data ETTh1 \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len ${pl0} \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp_OLinearlike' \
        --batch_size 32 \
        --itr 1 \
        --auxi_lambda 0 \
        --rec_lambda 1 \
        --fix_seed ${seed} \
        --checkpoints $CHECKPOINTS \
        --results $RESULTS \
        --test_results $TEST_RESULTS \
        --log_path $LOG_PATH \
        --iter_norm ${iter_norm} \
        --num_groups ${num_groups} \
        --T ${T} \
        --revin ${revin} \
        --momentum ${momentum} \
        --affine ${affine} \
        --d_model ${d_model0} \
        --add_module ${add_module} \
        --individual ${individual} \
        --learning_rate ${lr0} \
        --lradj type1 \
        --loss_mode L1 \
        --patience 8 \
        --train_epochs 30 \
        --dropout 0.2 \
        | sed -r "s/\\x1B\\[[0-9;]*[mGKHF]//g" >> $LOG_DIR/gpu_0.log 2>&1 &
    pid0=$!

    echo "Start experiment: pred_len=$pl1 on GPU 1"
    CUDA_VISIBLE_DEVICES=1 python -u run_IN.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path $DATA_ROOT/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id "${dataset}_96_${pl1}" \
        --model ${model_name} \
        --data ETTh1 \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len ${pl1} \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp_OLinearlike' \
        --batch_size 32 \
        --itr 1 \
        --auxi_lambda 0 \
        --rec_lambda 1 \
        --fix_seed ${seed} \
        --checkpoints $CHECKPOINTS \
        --results $RESULTS \
        --test_results $TEST_RESULTS \
        --log_path $LOG_PATH \
        --iter_norm ${iter_norm} \
        --num_groups ${num_groups} \
        --T ${T} \
        --revin ${revin} \
        --momentum ${momentum} \
        --affine ${affine} \
        --d_model ${d_model1} \
        --add_module ${add_module} \
        --individual ${individual} \
        --learning_rate ${lr1} \
        --lradj type1 \
        --loss_mode L1 \
        --patience 8 \
        --train_epochs 30 \
        --dropout 0.2 \
        | sed -r "s/\\x1B\\[[0-9;]*[mGKHF]//g" >> $LOG_DIR/gpu_1.log 2>&1 &
    pid1=$!

    # 等待本轮两个实验都结束
    wait $pid0
    wait $pid1
    echo "Finished experiments: pred_len=$pl0 on GPU 0, pred_len=$pl1 on GPU 1"
done

echo "All finished ..."

