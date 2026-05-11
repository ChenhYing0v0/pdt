#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="experiments/revision/r3_4_gnn_baselines/timefilter_exchange_multi_pred_len.json"
RUN_ID="${RUN_ID:-r3_4_timefilter_exchange_s2023}"
GPU_ID="${GPU:-0}"
DRY_RUN=0

usage() {
  cat >&2 <<EOF
usage: run_r3_4_timefilter_exchange.sh [--gpu ID] [--dry-run]

Environment:
  DATA_ROOT      dataset root containing exchange_rate/exchange_rate.csv
  OUTPUT_ROOT    output root passed through scripts/remote/run_manifest.sh
  RUN_ID         run id prefix, default: ${RUN_ID}
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

CMD=(bash "$ROOT/scripts/remote/run_manifest.sh" "$MANIFEST" "$RUN_ID" --gpu "$GPU_ID")
if [[ "$DRY_RUN" == "1" ]]; then
  CMD+=(--dry-run)
fi

"${CMD[@]}"
