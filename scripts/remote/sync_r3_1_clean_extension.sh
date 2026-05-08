#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_RESULTS_ROOT="${LOCAL_RESULTS_ROOT:-$ROOT/artifacts/runs}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-$ROOT/artifacts/revision/r3_1_noise_robustness_extension/clean_checkpoints}"
DATASET="all"

usage() {
  cat >&2 <<'EOF'
usage: sync_r3_1_clean_extension.sh [--dataset weather|ettm2|all]

Syncs Weather/ETTm2 clean-checkpoint runs from REMOTE_RESULTS_ROOT, archives
best.ckpt plus lightweight metadata, and rebuilds the clean checkpoint index.
Set REMOTE_HOST and optionally REMOTE_RESULTS_ROOT before running locally.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      if [[ -z "${2:-}" ]]; then
        echo "--dataset requires weather, ettm2, or all." >&2
        exit 1
      fi
      DATASET="$2"
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

RUN_IDS=()

append_dataset_runs() {
  local dataset="$1"
  local model horizon
  for model in pdt itransformer dlinear; do
    for horizon in 96 192 336 720; do
      RUN_IDS+=("r3_1_clean_${model}_${dataset}_s2023_pl${horizon}")
    done
  done
}

case "$DATASET" in
  weather)
    append_dataset_runs "weather"
    ;;
  ettm2)
    append_dataset_runs "ettm2"
    ;;
  all)
    append_dataset_runs "weather"
    append_dataset_runs "ettm2"
    ;;
  *)
    echo "--dataset must be weather, ettm2, or all; got: $DATASET" >&2
    exit 1
    ;;
esac

mkdir -p "$ARCHIVE_ROOT"

echo "== Syncing R3.1 extension clean checkpoint runs =="
LOCAL_RESULTS_ROOT="$LOCAL_RESULTS_ROOT" bash "$ROOT/scripts/remote/sync_results.sh" "${RUN_IDS[@]}"

INDEX_PATH="$ARCHIVE_ROOT/index.tsv"
printf "run_id\tcheckpoint\tmetrics\tconfig\tgit_meta\ttrain_log\n" > "$INDEX_PATH"

for run_id in "${RUN_IDS[@]}"; do
  run_dir="$LOCAL_RESULTS_ROOT/$run_id"
  archive_dir="$ARCHIVE_ROOT/$run_id"
  mkdir -p "$archive_dir"

  checkpoint="$run_dir/best.ckpt"
  if [[ ! -f "$checkpoint" ]]; then
    if [[ -d "$run_dir/checkpoints" ]]; then
      checkpoint="$(find "$run_dir/checkpoints" -path "*/checkpoint.pth" -type f | head -n 1 || true)"
    else
      checkpoint=""
    fi
  fi
  if [[ -z "$checkpoint" || ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint for ${run_id}" >&2
    exit 1
  fi

  cp "$checkpoint" "$archive_dir/best.ckpt"
  for name in metrics.json config.snapshot.json git_meta.json train.log; do
    if [[ -f "$run_dir/$name" ]]; then
      cp "$run_dir/$name" "$archive_dir/$name"
    fi
  done

  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$run_id" \
    "$archive_dir/best.ckpt" \
    "$archive_dir/metrics.json" \
    "$archive_dir/config.snapshot.json" \
    "$archive_dir/git_meta.json" \
    "$archive_dir/train.log" >> "$INDEX_PATH"
done

echo "Archived clean checkpoints under: $ARCHIVE_ROOT"
echo "Index: $INDEX_PATH"
