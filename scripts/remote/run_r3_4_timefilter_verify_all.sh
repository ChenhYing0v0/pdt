#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_ID="${GPU:-0}"
DRY_RUN=0
RUN_PREFIX="${RUN_PREFIX:-r3_4_timefilter_verify}"

usage() {
  cat >&2 <<EOF
usage: run_r3_4_timefilter_verify_all.sh [--gpu ID] [--dry-run]

Environment:
  DATA_ROOT      dataset root containing ETT-small/, electricity/, weather/, exchange_rate/, traffic/
  OUTPUT_ROOT    output root passed through scripts/remote/run_manifest.sh
  RUN_PREFIX     run id prefix, default: ${RUN_PREFIX}
  GPU            default GPU id if --gpu is omitted, default: ${GPU_ID}
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
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

MANIFESTS=(
  "experiments/revision/r3_4_gnn_baselines/timefilter_ettm1_multi_pred_len.json"
  "experiments/revision/r3_4_gnn_baselines/timefilter_ettm2_multi_pred_len.json"
  "experiments/revision/r3_4_gnn_baselines/timefilter_etth1_multi_pred_len.json"
  "experiments/revision/r3_4_gnn_baselines/timefilter_etth2_multi_pred_len.json"
  "experiments/revision/r3_4_gnn_baselines/timefilter_ecl_multi_pred_len.json"
  "experiments/revision/r3_4_gnn_baselines/timefilter_weather_multi_pred_len.json"
  "experiments/revision/r3_4_gnn_baselines/timefilter_exchange_multi_pred_len.json"
  "experiments/revision/r3_4_gnn_baselines/timefilter_traffic_multi_pred_len.json"
)

for manifest in "${MANIFESTS[@]}"; do
  dataset="$(basename "$manifest" _multi_pred_len.json)"
  dataset="${dataset#timefilter_}"
  run_id="${RUN_PREFIX}_${dataset}_s2023"
  cmd=(bash "$ROOT/scripts/remote/run_manifest.sh" "$manifest" "$run_id" --gpu "$GPU_ID")
  if [[ "$DRY_RUN" == "1" ]]; then
    cmd+=(--dry-run)
  fi
  echo "==> ${cmd[*]}"
  "${cmd[@]}"
done
