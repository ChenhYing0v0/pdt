#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST_DIR="$ROOT/experiments/revision/r3_2_hyperparam_sensitivity"
RUNNER="$ROOT/scripts/remote/run_manifest.sh"

usage() {
  cat >&2 <<EOF
usage: run_r3_2_hyperparam_sensitivity.sh [--dataset NAME] [--pred-len N] [--gpu ID] [--dry-run] [--skip-predictions]

Runs the standard R3.2/R1.4 PDT hyperparameter sensitivity manifests:
  - ETTh1, pred_len=336
  - Weather, pred_len=336
  - Top-K grid: 8, 16, 32, 64, 96
  - Mask threshold grid: 0.05, 0.10, 0.15, 0.25

Filters:
  --dataset NAME    one of: all, etth1, weather (default: all)
  --pred-len N      one of: all, 336 (default: all)
EOF
}

EXTRA_ARGS=()
DATASET_FILTER="all"
PRED_LEN_FILTER="all"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      if [[ -z "${2:-}" ]]; then
        echo "--dataset requires a value." >&2
        exit 1
      fi
      DATASET_FILTER="$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')"
      shift 2
      ;;
    --pred-len|--pred_len)
      if [[ -z "${2:-}" ]]; then
        echo "--pred-len requires a value." >&2
        exit 1
      fi
      PRED_LEN_FILTER="$2"
      shift 2
      ;;
    --gpu)
      if [[ -z "${2:-}" ]]; then
        echo "--gpu requires a GPU id." >&2
        exit 1
      fi
      EXTRA_ARGS+=(--gpu "$2")
      shift 2
      ;;
    --dry-run|--skip-predictions)
      EXTRA_ARGS+=("$1")
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

if [[ ! -d "$MANIFEST_DIR" ]]; then
  echo "Manifest directory not found: $MANIFEST_DIR" >&2
  echo "Run scripts/revision/generate_r3_2_hyperparam_sensitivity_manifests.py first." >&2
  exit 1
fi

case "$DATASET_FILTER" in
  all|etth1|weather) ;;
  *)
    echo "Unsupported dataset filter: $DATASET_FILTER" >&2
    usage
    exit 1
    ;;
esac

case "$PRED_LEN_FILTER" in
  all|336) ;;
  *)
    echo "Unsupported pred-len filter: $PRED_LEN_FILTER" >&2
    usage
    exit 1
    ;;
esac

selected=0
for manifest in "$MANIFEST_DIR"/*.json; do
  name="$(basename "$manifest" .json)"
  manifest_dataset="${name%%_pl*}"
  manifest_pred_len="$(printf '%s\n' "$name" | sed -E 's/^.*_pl([0-9]+)_.*$/\1/')"

  if [[ "$DATASET_FILTER" != "all" && "$manifest_dataset" != "$DATASET_FILTER" ]]; then
    continue
  fi
  if [[ "$PRED_LEN_FILTER" != "all" && "$manifest_pred_len" != "$PRED_LEN_FILTER" ]]; then
    continue
  fi

  selected=$((selected + 1))
  "$RUNNER" "$manifest" "r3_2_${name}" "${EXTRA_ARGS[@]}"
done

if [[ "$selected" -eq 0 ]]; then
  echo "No manifests matched dataset=$DATASET_FILTER pred_len=$PRED_LEN_FILTER." >&2
  exit 1
fi
