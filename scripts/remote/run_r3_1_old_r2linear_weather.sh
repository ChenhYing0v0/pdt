#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pdt}"
GPU_ID="${GPU:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/exp_outputs/r-2026-pdt}"
PDT_OLD_ROOT="${PDT_OLD_ROOT:-$ROOT/baselines/PDT_old}"
OLD_DATA_ROOT="${OLD_DATA_ROOT:-$ROOT/baselines/PDT/dataset}"
OLD_RUN_IN="${OLD_RUN_IN:-$PDT_OLD_ROOT/run_IN.py}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-r3_1_old_r2linear_weather_s2023}"
PRED_LENS="96,192,336,720"
DRY_RUN=0
ONLY_MISSING=0

usage() {
  cat >&2 <<'EOF'
usage: run_r3_1_old_r2linear_weather.sh [--gpu ID] [--pred-lens 96,192,336,720] [--dry-run] [--only-missing]

Runs the old Weather R2Linear configuration in the same remote environment for
R3.1 debugging. The parameter schedule matches old/R2Linear_weather.sh. The
process runs from baselines/PDT_old and calls baselines/PDT_old/run_IN.py to
match the manually verified old-version execution context.

Environment:
  CONDA_ENV_NAME  conda env name, default: pdt
  OUTPUT_ROOT     run output root, default: $HOME/exp_outputs/r-2026-pdt
  OLD_DATA_ROOT   Weather csv and mats root, default: <repo>/baselines/PDT/dataset
  PDT_OLD_ROOT    old baseline root, default: <repo>/baselines/PDT_old
  OLD_RUN_IN      launcher script, default: <repo>/baselines/PDT_old/run_IN.py
  RUN_ID_PREFIX   output run id prefix, default: r3_1_old_r2linear_weather_s2023
EOF
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

if [[ ! -f "$OLD_RUN_IN" ]]; then
  echo "Missing old run_IN.py: $OLD_RUN_IN" >&2
  exit 1
fi
if [[ ! -d "$PDT_OLD_ROOT" ]]; then
  echo "Missing PDT_old root: $PDT_OLD_ROOT" >&2
  exit 1
fi
if [[ ! -f "$OLD_DATA_ROOT/weather/weather.csv" ]]; then
  echo "Missing old Weather data: $OLD_DATA_ROOT/weather/weather.csv" >&2
  exit 1
fi

if command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run --no-capture-output -n "$CONDA_ENV_NAME" python -u)
else
  PYTHON_CMD=(python -u)
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

  local run_id="${RUN_ID_PREFIX}_pl${pred_len}"
  local run_dir="$OUTPUT_ROOT/$run_id"
  local checkpoints="$run_dir/checkpoints"
  local results="$run_dir/results"
  local test_results="$run_dir/test_results"
  local log_path="$run_dir/result_long_term_forecast.txt"
  local train_log="$run_dir/train.log"

  local cmd=(
    "${PYTHON_CMD[@]}" "$OLD_RUN_IN"
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
    --target OT
    --freq h
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
    --num_workers 8
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

  echo "== Running $run_id =="
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'run_id=%s\n' "$run_id"
    printf 'cwd=%s\n' "$PDT_OLD_ROOT"
    printf 'OLD_RUN_IN=%s\n' "$OLD_RUN_IN"
    printf 'PDT_OLD_ROOT=%s\n' "$PDT_OLD_ROOT"
    printf 'OLD_DATA_ROOT=%s\n' "$OLD_DATA_ROOT"
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "$GPU_ID"
    printf 'PYTHONPATH=%s\n' "$PDT_OLD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    printf 'command='
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return
  fi

  mkdir -p "$checkpoints" "$results" "$test_results"
  if [[ "$ONLY_MISSING" -eq 1 ]] && find "$results" -name metrics.npy -type f -print -quit | grep -q .; then
    echo "skip existing $run_id"
    return
  fi

  {
    printf 'run_id=%s\n' "$run_id"
    printf 'cwd=%s\n' "$PDT_OLD_ROOT"
    printf 'OLD_RUN_IN=%s\n' "$OLD_RUN_IN"
    printf 'PDT_OLD_ROOT=%s\n' "$PDT_OLD_ROOT"
    printf 'OLD_DATA_ROOT=%s\n' "$OLD_DATA_ROOT"
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "$GPU_ID"
    printf 'PYTHONPATH=%s\n' "$PDT_OLD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    printf 'command='
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } > "$run_dir/command.txt"

  (
    cd "$PDT_OLD_ROOT"
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    export PYTHONPATH="$PDT_OLD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    "${cmd[@]}"
  ) 2>&1 | sed -r "s/\\x1B\\[[0-9;]*[mGKHF]//g" | tee "$train_log"
}

for pred_len in "${REQUESTED_PRED_LENS[@]}"; do
  pred_len="${pred_len//[[:space:]]/}"
  if [[ -z "$pred_len" ]]; then
    continue
  fi
  run_one "$pred_len"
done
