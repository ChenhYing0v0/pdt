#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_ID="${GPU:-0}"
PDT_OLD_ROOT="${PDT_OLD_ROOT:-$ROOT/baselines/PDT_old}"
OLD_DATA_ROOT="${OLD_DATA_ROOT:-$ROOT/baselines/PDT/dataset}"
OLD_OUTPUT_DIR="${OLD_OUTPUT_DIR:-./exp_results/R2Linear}"
PRED_LENS="96,192,336,720"
DRY_RUN=0
ONLY_MISSING=0

usage() {
  cat >&2 <<'EOF_USAGE'
usage: run_r3_1_old_r2linear_weather.sh [--gpu ID] [--pred-lens 96,192,336,720] [--dry-run] [--only-missing]

Runs Weather R2Linear through the native old-version execution path. This script
intentionally mirrors the manually verified workflow:

  cd baselines/PDT_old
  CUDA_VISIBLE_DEVICES=<gpu> python -u run_IN.py ... | sed ... >> log 2>&1

The fixed parameter schedule matches old/R2Linear_weather.sh. Data and matrices
are read from baselines/PDT/dataset by default, matching the successful manual
reproduction after DATA_ROOT was changed to the PDT dataset root.

Environment:
  GPU             GPU id, default: 0
  OLD_DATA_ROOT   Weather csv and mats root, default: <repo>/baselines/PDT/dataset
  PDT_OLD_ROOT    old baseline root, default: <repo>/baselines/PDT_old
  OLD_OUTPUT_DIR  output directory relative to PDT_OLD_ROOT unless absolute,
                  default: ./exp_results/R2Linear
EOF_USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      if [[ -z "${2:-}" ]]; then
        echo "--gpu requires a GPU id." >&2
        exit 1
      fi
      GPU_ID="$2"
      shift 2
      ;;
    --pred-lens)
      if [[ -z "${2:-}" ]]; then
        echo "--pred-lens requires a comma-separated list." >&2
        exit 1
      fi
      PRED_LENS="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --only-missing)
      ONLY_MISSING=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "$PDT_OLD_ROOT" ]]; then
  echo "Missing PDT_old root: $PDT_OLD_ROOT" >&2
  exit 1
fi
if [[ ! -f "$PDT_OLD_ROOT/run_IN.py" ]]; then
  echo "Missing old run_IN.py: $PDT_OLD_ROOT/run_IN.py" >&2
  exit 1
fi
if [[ ! -f "$OLD_DATA_ROOT/weather/weather.csv" ]]; then
  echo "Missing Weather data: $OLD_DATA_ROOT/weather/weather.csv" >&2
  exit 1
fi

IFS=',' read -r -a REQUESTED_PRED_LENS <<< "$PRED_LENS"

schedule_for_pred_len() {
  local pred_len="$1"
  case "$pred_len" in
    96)
      R_RANK=96
      K_TOP=8
      ALPHA_INIT=0.1
      MASK_THRESHOLD=0.05
      ;;
    192)
      R_RANK=96
      K_TOP=96
      ALPHA_INIT=0.4
      MASK_THRESHOLD=0.1
      ;;
    336)
      R_RANK=96
      K_TOP=64
      ALPHA_INIT=0.25
      MASK_THRESHOLD=0.25
      ;;
    720)
      R_RANK=96
      K_TOP=8
      ALPHA_INIT=0.05
      MASK_THRESHOLD=0.05
      ;;
    *)
      echo "Unsupported pred_len for old/R2Linear_weather.sh schedule: $pred_len" >&2
      exit 1
      ;;
  esac
}

run_one() {
  local pred_len="$1"
  schedule_for_pred_len "$pred_len"

  local job_dir="$OLD_OUTPUT_DIR/R2Linear_weather"
  local log_dir="$job_dir/log"
  local checkpoints="$job_dir/checkpoints/"
  local results="$job_dir/results/"
  local test_results="$job_dir/test_results/"
  local log_path="$job_dir/result_long_term_forecast.txt"
  local pred_log="$log_dir/pred_len_${pred_len}.log"
  local setting="long_term_forecast_weather_96_${pred_len}_R2Linear_custom_ftM_sl96_ll48_pl${pred_len}_dm512_nh8_el2_dl1_df512_fc3_ebtimeF_dtTrue_rr${R_RANK}_frR0_k${K_TOP}_Exp_0.1a_Rk_Dnorm_k-EU-mask"
  local output_abs="$OLD_OUTPUT_DIR"
  if [[ "$OLD_OUTPUT_DIR" != /* ]]; then
    output_abs="$PDT_OLD_ROOT/$OLD_OUTPUT_DIR"
  fi

  local cmd=(
    python -u run_IN.py
    --task_name long_term_forecast
    --is_training 1
    --root_path "$OLD_DATA_ROOT/weather/"
    --data_path weather.csv
    --q_mat_file "$OLD_DATA_ROOT/RRR_mats/weather/weather_RRR_L96_R${R_RANK}_H${pred_len}_Qin.npy"
    --q_out_mat_file "$OLD_DATA_ROOT/PCCA_mats/weather/weather_PCCA_OUT_H${pred_len}_identity_avg.npy"
    --r_mat_file "$OLD_DATA_ROOT/RRR_mats/weather/weather_RRR_L96_R${R_RANK}_H${pred_len}_R.npy"
    --rk_mat_file "$OLD_DATA_ROOT/RRR_mats/weather/weather_RRR_L96_R${K_TOP}_H${pred_len}_R.npy"
    --Q_chan_indep 0
    --model_id "weather_96_${pred_len}"
    --model R2Linear
    --data custom
    --features M
    --seq_len 96
    --label_len 48
    --pred_len "$pred_len"
    --e_layers 2
    --d_layers 1
    --factor 3
    --enc_in 21
    --dec_in 21
    --c_out 21
    --des "Exp_0.1a_Rk_Dnorm_k-EU-mask"
    --embed_size 16
    --d_model 512
    --d_ff 512
    --batch_size 32
    --itr 1
    --auxi_lambda 0
    --rec_lambda 1
    --fix_seed 2023
    --checkpoints "$checkpoints"
    --results "$results"
    --test_results "$test_results"
    --log_path "$log_path"
    --learning_rate 1e-3
    --lradj type3
    --loss_mode L1
    --patience 5
    --train_epochs 30
    --CKA_flag 0
    --dropout 0.0
    --freeze_R 0
    --r_rank "$R_RANK"
    --k_top "$K_TOP"
    --alpha_init "$ALPHA_INIT"
    --mask_threshold "$MASK_THRESHOLD"
  )

  echo "Start experiment: pred_len=$pred_len"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'cwd=%s\n' "$PDT_OLD_ROOT"
    printf 'OLD_DATA_ROOT=%s\n' "$OLD_DATA_ROOT"
    printf 'OLD_OUTPUT_DIR=%s\n' "$OLD_OUTPUT_DIR"
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "$GPU_ID"
    printf 'log=%s\n' "$pred_log"
    printf 'command=CUDA_VISIBLE_DEVICES=%q ' "$GPU_ID"
    printf '%q ' "${cmd[@]}"
    printf '| sed -r %q >> %q 2>&1\n' 's/\x1B\[[0-9;]*[mGKHF]//g' "$pred_log"
    return
  fi

  mkdir -p "$output_abs/R2Linear_weather/log" \
    "$output_abs/R2Linear_weather/checkpoints" \
    "$output_abs/R2Linear_weather/results" \
    "$output_abs/R2Linear_weather/test_results"
  if [[ "$ONLY_MISSING" -eq 1 ]] && [[ -f "$output_abs/R2Linear_weather/results/$setting/metrics.npy" ]]; then
    echo "skip existing pred_len=$pred_len: $output_abs/R2Linear_weather/results/$setting/metrics.npy"
    return
  fi

  (
    cd "$PDT_OLD_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU_ID" "${cmd[@]}" \
      | sed -r "s/\\x1B\\[[0-9;]*[mGKHF]//g" >> "$pred_log" 2>&1
  )
  echo "Finished experiment: pred_len=$pred_len"
}

echo "R2Linear on weather"
echo "PDT_OLD_ROOT: $PDT_OLD_ROOT"
echo "OLD_DATA_ROOT: $OLD_DATA_ROOT"
echo "OLD_OUTPUT_DIR: $OLD_OUTPUT_DIR"
echo "Start time: $(date)"

for pred_len in "${REQUESTED_PRED_LENS[@]}"; do
  pred_len="${pred_len//[[:space:]]/}"
  if [[ -z "$pred_len" ]]; then
    continue
  fi
  run_one "$pred_len"
done

echo "All finished ..."
echo "End time: $(date)"
