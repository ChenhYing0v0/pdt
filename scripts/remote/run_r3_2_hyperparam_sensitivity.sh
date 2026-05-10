#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST_DIR="$ROOT/experiments/revision/r3_2_hyperparam_sensitivity"
RUNNER="$ROOT/scripts/remote/run_manifest.sh"

usage() {
  cat >&2 <<EOF
usage: run_r3_2_hyperparam_sensitivity.sh [--gpu ID] [--dry-run] [--skip-predictions]

Runs the standard R3.2/R1.4 PDT hyperparameter sensitivity manifests:
  - ETTh1, pred_len=336
  - Weather, pred_len=336
  - Top-K grid: 8, 16, 32, 64, 96
  - Mask threshold grid: 0.05, 0.10, 0.15, 0.25
EOF
}

EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
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

for manifest in "$MANIFEST_DIR"/*.json; do
  name="$(basename "$manifest" .json)"
  "$RUNNER" "$manifest" "r3_2_${name}" "${EXTRA_ARGS[@]}"
done
