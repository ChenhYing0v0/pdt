#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pdt}"
GPU_ID="${GPU:-}"

usage() {
  cat >&2 <<'EOF'
usage: run_r3_1_corruption_eval.sh [--gpu ID] [--dry-run] [--collect-only]

Runs Reviewer #3.1 ETTh1 test-time spike/segment corruption evaluation from
the synced clean checkpoint index.
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
      GPU_ID="$2"
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

CMD=(python -u "$ROOT/scripts/revision/eval_r3_1_noise_robustness.py" "${EXTRA_ARGS[@]}")
if [[ -n "$GPU_ID" ]]; then
  CMD+=(--gpu "$GPU_ID")
fi

cd "$ROOT"
if command -v conda >/dev/null 2>&1; then
  conda run --no-capture-output -n "$CONDA_ENV_NAME" "${CMD[@]}"
else
  "${CMD[@]}"
fi
