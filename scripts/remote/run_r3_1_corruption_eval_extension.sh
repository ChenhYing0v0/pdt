#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_ID="${GPU:-}"
DATASET="all"
CLEAN_INDEX="${CLEAN_INDEX:-artifacts/revision/r3_1_noise_robustness_extension/clean_checkpoints/index.tsv}"
CLEAN_RUNS_ROOT="${CLEAN_RUNS_ROOT:-$HOME/exp_outputs/r-2026-pdt}"
CORRUPTION_OUTPUT_ROOT="${CORRUPTION_OUTPUT_ROOT:-artifacts/revision/r3_1_noise_robustness_extension}"
RUN_GLOB="${RUN_GLOB:-}"
EXTRA_ARGS=()

usage() {
  cat >&2 <<'EOF'
usage: run_r3_1_corruption_eval_extension.sh [--gpu ID] [--dataset weather|ettm2|all] [--run-glob GLOB] [--dry-run] [--collect-only] [--only-missing]

Runs Reviewer #3.1 spike/segment corruption evaluation for the Weather/ETTm2
extension from the clean checkpoint index. Execute this on the remote machine
after clean checkpoints are trained and indexed. Activate the intended Python
environment before running this script. Set CORRUPTION_OUTPUT_ROOT to override
where corruption-evaluation artifacts are written.
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
    --run-glob)
      if [[ -z "${2:-}" ]]; then
        echo "--run-glob requires a glob pattern." >&2
        exit 1
      fi
      RUN_GLOB="$2"
      shift 2
      ;;
    --dry-run|--collect-only|--only-missing)
      EXTRA_ARGS+=("$1")
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

case "$DATASET" in
  weather)
    DATASETS="Weather"
    RUN_GLOB="${RUN_GLOB:-r3_1_clean_*_weather_s2023_pl*}"
    ;;
  ettm2)
    DATASETS="ETTm2"
    RUN_GLOB="${RUN_GLOB:-r3_1_clean_*_ettm2_s2023_pl*}"
    ;;
  all)
    DATASETS="Weather,ETTm2"
    RUN_GLOB="${RUN_GLOB:-r3_1_clean_*_s2023_pl*}"
    ;;
  *)
    echo "--dataset must be weather, ettm2, or all; got: $DATASET" >&2
    exit 1
    ;;
esac

CMD=(
  python -u "$ROOT/scripts/revision/eval_r3_1_noise_robustness.py"
  --clean-index "$CLEAN_INDEX"
  --clean-runs-root "$CLEAN_RUNS_ROOT"
  --run-glob "$RUN_GLOB"
  --output-root "$CORRUPTION_OUTPUT_ROOT"
  --datasets "$DATASETS"
  "${EXTRA_ARGS[@]}"
)
if [[ -n "$GPU_ID" ]]; then
  CMD+=(--gpu "$GPU_ID")
fi

cd "$ROOT"
"${CMD[@]}"
