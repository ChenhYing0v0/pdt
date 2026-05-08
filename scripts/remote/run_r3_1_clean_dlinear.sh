#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_ID="${GPU:-}"

usage() {
  cat >&2 <<'EOF'
usage: run_r3_1_clean_dlinear.sh [--gpu ID]

Runs only the DLinear clean-checkpoint manifest for Reviewer #3.1 robustness
experiments.
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

cmd=(bash "$ROOT/scripts/remote/run_manifest.sh" \
  "experiments/revision/r3_1_noise_robustness/clean_checkpoints/dlinear_etth1_clean.json" \
  "r3_1_clean_dlinear_etth1_s2023" \
  --skip-predictions)
if [[ -n "$GPU_ID" ]]; then
  cmd+=(--gpu "$GPU_ID")
fi

echo "== Running r3_1_clean_dlinear_etth1_s2023 =="
"${cmd[@]}"
