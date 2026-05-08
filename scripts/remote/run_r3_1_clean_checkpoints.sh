#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_ID="${GPU:-}"

usage() {
  cat >&2 <<'EOF'
usage: run_r3_1_clean_checkpoints.sh [--gpu ID]

Runs the remaining clean-checkpoint training manifests for Reviewer #3.1
robustness experiments. PDT has already completed, so this entry now launches
only iTransformer and DLinear. Execute this on the remote training machine from
the repository checkout. Set DATA_ROOT, CONDA_ENV_NAME, and OUTPUT_ROOT as
needed.
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

run_manifest() {
  local manifest="$1"
  local run_id="$2"
  local cmd=(bash "$ROOT/scripts/remote/run_manifest.sh" "$manifest" "$run_id" --skip-predictions)
  if [[ -n "$GPU_ID" ]]; then
    cmd+=(--gpu "$GPU_ID")
  fi
  echo "== Running ${run_id} =="
  "${cmd[@]}"
}

run_manifest "experiments/revision/r3_1_noise_robustness/clean_checkpoints/itransformer_etth1_clean.json" \
  "r3_1_clean_itransformer_etth1_s2023"
run_manifest "experiments/revision/r3_1_noise_robustness/clean_checkpoints/dlinear_etth1_clean.json" \
  "r3_1_clean_dlinear_etth1_s2023"
