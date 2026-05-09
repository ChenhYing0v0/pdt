#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_ID="${GPU:-}"
DATASET="all"
DRY_RUN=0

usage() {
  cat >&2 <<'EOF'
usage: run_r3_1_clean_extension.sh [--gpu ID] [--dataset weather|ettm2|all] [--dry-run]

Runs clean-checkpoint training for the Reviewer #3.1 robustness extension on
Weather and ETTm2. Each manifest expands to pred_len 96/192/336/720. Execute
this on the remote training machine from the repository checkout. Activate the
intended environment first, and set DATA_ROOT and OUTPUT_ROOT as needed.
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
    --dataset)
      if [[ -z "${2:-}" ]]; then
        echo "--dataset requires weather, ettm2, or all." >&2
        exit 1
      fi
      DATASET="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
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

run_manifest() {
  local manifest="$1"
  local run_id="$2"
  local skip_predictions="${3:-1}"
  local cmd=(bash "$ROOT/scripts/remote/run_manifest.sh" "$manifest" "$run_id")
  if [[ "$skip_predictions" -eq 1 ]]; then
    cmd+=(--skip-predictions)
  fi
  if [[ -n "$GPU_ID" ]]; then
    cmd+=(--gpu "$GPU_ID")
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    cmd+=(--dry-run)
  fi
  echo "== Running ${run_id} =="
  "${cmd[@]}"
}

run_dataset() {
  local dataset="$1"
  local base="experiments/revision/r3_1_noise_robustness/clean_checkpoints_extension"
  run_manifest "$base/pdt_${dataset}_clean.json" "r3_1_clean_pdt_${dataset}_s2023"
  run_manifest "$base/itransformer_${dataset}_clean.json" "r3_1_clean_itransformer_${dataset}_s2023"
  run_manifest "$base/dlinear_${dataset}_clean.json" "r3_1_clean_dlinear_${dataset}_s2023"
}

case "$DATASET" in
  weather)
    run_dataset "weather"
    ;;
  ettm2)
    run_dataset "ettm2"
    ;;
  all)
    run_dataset "weather"
    run_dataset "ettm2"
    ;;
  *)
    echo "--dataset must be weather, ettm2, or all; got: $DATASET" >&2
    exit 1
    ;;
esac
