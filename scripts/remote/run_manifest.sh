#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/exp_outputs/r-2026-pdt}"
NOHUP_LOG_ROOT="${NOHUP_LOG_ROOT:-$OUTPUT_ROOT/_nohup}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pdt}"

usage() {
  cat >&2 <<EOF
usage: run_manifest.sh <manifest> [run_id] [--gpu ID] [--nohup] [--dry-run] [--skip-predictions]

Environment:
  CONDA_ENV_NAME   conda env name, default: ${CONDA_ENV_NAME}
  OUTPUT_ROOT      output root, default: ${OUTPUT_ROOT}
  NOHUP_LOG_ROOT   nohup log root, default: ${NOHUP_LOG_ROOT}
  GPU              default GPU id if --gpu is omitted; if unset, keep manifest env
EOF
}

MANIFEST=""
RUN_ID=""
GPU_ID="${GPU:-}"
USE_NOHUP=0
DRY_RUN=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nohup)
      USE_NOHUP=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      EXTRA_ARGS+=("$1")
      shift
      ;;
    --gpu)
      if [[ -z "${2:-}" ]]; then
        echo "--gpu requires a GPU id." >&2
        exit 1
      fi
      GPU_ID="$2"
      shift 2
      ;;
    --skip-predictions)
      EXTRA_ARGS+=(--set args.skip_predictions=true)
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --*)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      if [[ -z "$MANIFEST" ]]; then
        MANIFEST="$1"
      elif [[ -z "$RUN_ID" ]]; then
        RUN_ID="$1"
      else
        echo "Unexpected positional argument: $1" >&2
        usage
        exit 1
      fi
      shift
      ;;
  esac
done

if [[ -z "$MANIFEST" ]]; then
  usage
  exit 1
fi

if command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run --no-capture-output -n "$CONDA_ENV_NAME" python -u)
elif [[ "$DRY_RUN" -eq 1 ]] && command -v python3 >/dev/null 2>&1; then
  echo "conda command not found; using python3 for dry-run only." >&2
  PYTHON_CMD=(python3 -u)
else
  echo "conda command not found; cannot guarantee env '${CONDA_ENV_NAME}'." >&2
  exit 1
fi

cd "$ROOT"
CMD=("${PYTHON_CMD[@]}" -m protocol.runners.run_manifest --manifest "$MANIFEST" --output-root "$OUTPUT_ROOT")
if [[ -n "$RUN_ID" ]]; then
  CMD+=(--run-id "$RUN_ID")
fi
if [[ -n "$GPU_ID" ]]; then
  CMD+=(--set "env.CUDA_VISIBLE_DEVICES=$GPU_ID")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

if [[ "$USE_NOHUP" -eq 1 ]]; then
  mkdir -p "$NOHUP_LOG_ROOT"
  stamp="$(date +"%Y%m%d_%H%M%S")"
  manifest_name="$(basename "$MANIFEST" .json)"
  gpu_suffix=""
  if [[ -n "$GPU_ID" ]]; then
    gpu_suffix="_gpu${GPU_ID}"
  fi
  log_path="$NOHUP_LOG_ROOT/${stamp}_${manifest_name}${gpu_suffix}.log"
  nohup "${CMD[@]}" >"$log_path" 2>&1 &
  pid=$!
  echo "Started in background."
  echo "PID: $pid"
  echo "Wrapper log: $log_path"
  echo "Output root: $OUTPUT_ROOT"
  echo "Conda env: $CONDA_ENV_NAME"
  if [[ -n "$GPU_ID" ]]; then
    echo "GPU: $GPU_ID"
  fi
  exit 0
fi

"${CMD[@]}"
