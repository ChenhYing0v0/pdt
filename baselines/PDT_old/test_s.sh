export CUDA_VISIBLE_DEVICES=0

# model_name=BDLinear


# python -u run_IN.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_96_96 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --auxi_lambda 0 \
#   --rec_lambda 1 \
#   --d_model 512 \
#   --iter_norm 1 \
#   --norm_type ind \
#   --num_groups 1 \
#   --T 3 \
#   --revin 0 \
#   --momentum 0.01 \
#   --affine 0 \
#   --individual 0 \
#   --save_cov 0 \
#   --norm_type ind \
#   --itr 1



model_name=DLinear  


python -u run_IN.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --q_mat_file ./dataset/cov_mats/ETTh1/ETTh1_96_ratio0.6.npy\
  --q_out_mat_file ./dataset/cov_mats/ETTh1/ETTh1_96_ratio0.6.npy\
  --Q_MAT_file ./dataset/cov_mats/ETTh1/channel_corr_mat/ETTh1_COV_channel_ratio0.60.npy \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --auxi_lambda 0 \
  --rec_lambda 1 \
  --embed_size 16 \
  --d_model 512 \
  --d_ff 512 \
  --CKA_flag 0 \
  --dropout 0.2 \
  --itr 1