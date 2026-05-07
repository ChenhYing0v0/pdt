#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: sync_results.sh [run_id ...] [--all] [--lite]

  --all   sync all runs under the remote results root
  --lite  skip large artifacts (*.npz and checkpoints/)
EOF
}

SYNC_ALL=0
SYNC_LITE=0
RUN_IDS=()

build_rsync_args() {
  RSYNC_ARGS=(-av)
  if [[ "$SYNC_LITE" -eq 1 ]]; then
    RSYNC_ARGS+=(--exclude "*.npz" --exclude "checkpoints/")
  fi
}

ensure_remote_root_exists() {
  if [[ "$REMOTE_RESULTS_ROOT" == /Users/* ]]; then
    echo "REMOTE_RESULTS_ROOT looks like a local macOS path: ${REMOTE_RESULTS_ROOT}" >&2
    echo "Use a quoted remote path like '~/exp_outputs/r-2026-pdt' or an absolute Linux path." >&2
    exit 1
  fi
  if ! ssh "$REMOTE_HOST" "test -d ${remote_check_root}"; then
    echo "Remote results root does not exist: ${REMOTE_RESULTS_ROOT}" >&2
    echo "Hint: REMOTE_RESULTS_ROOT should be the path on the remote Linux host." >&2
    echo "Use a quoted '~' path like '~/exp_outputs/r-2026-pdt' or an absolute Linux path." >&2
    exit 1
  fi
}

sync_one() {
  local run_id="$1"
  if ! ssh "$REMOTE_HOST" "test -d ${remote_check_root}/${run_id}"; then
    echo "Remote run directory does not exist: ${REMOTE_RESULTS_ROOT}/${run_id}" >&2
    exit 1
  fi
  build_rsync_args
  rsync "${RSYNC_ARGS[@]}" "${REMOTE_HOST}:${REMOTE_RESULTS_ROOT}/${run_id}/" "${LOCAL_RESULTS_ROOT}/${run_id}/"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      SYNC_ALL=1
      ;;
    --lite)
      SYNC_LITE=1
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
      RUN_IDS+=("$1")
      ;;
  esac
  shift
done

REMOTE_HOST="${REMOTE_HOST:?set REMOTE_HOST before syncing}"
REMOTE_RESULTS_ROOT="${REMOTE_RESULTS_ROOT:-~/exp_outputs/r-2026-pdt}"
LOCAL_RESULTS_ROOT="${LOCAL_RESULTS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/artifacts/runs}"
remote_check_root="${REMOTE_RESULTS_ROOT/#\~/\$HOME}"

mkdir -p "$LOCAL_RESULTS_ROOT"

ensure_remote_root_exists

if [[ ${#RUN_IDS[@]} -eq 0 || "$SYNC_ALL" -eq 1 ]]; then
  build_rsync_args
  rsync "${RSYNC_ARGS[@]}" "${REMOTE_HOST}:${REMOTE_RESULTS_ROOT}/" "${LOCAL_RESULTS_ROOT}/"
  exit 0
fi

for run_id in "${RUN_IDS[@]}"; do
  sync_one "$run_id"
done
