#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_RESULTS_ROOT="${LOCAL_RESULTS_ROOT:-$ROOT/artifacts/runs}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-$ROOT/artifacts/revision/r3_1_noise_robustness/clean_checkpoints}"

RUN_IDS=(
  r3_1_clean_pdt_etth1_s2023_pl96
  r3_1_clean_pdt_etth1_s2023_pl192
  r3_1_clean_pdt_etth1_s2023_pl336
  r3_1_clean_pdt_etth1_s2023_pl720
  r3_1_clean_itransformer_etth1_s2023_pl96
  r3_1_clean_itransformer_etth1_s2023_pl192
  r3_1_clean_itransformer_etth1_s2023_pl336
  r3_1_clean_itransformer_etth1_s2023_pl720
  r3_1_clean_dlinear_etth1_s2023_pl96
  r3_1_clean_dlinear_etth1_s2023_pl192
  r3_1_clean_dlinear_etth1_s2023_pl336
  r3_1_clean_dlinear_etth1_s2023_pl720
)

mkdir -p "$ARCHIVE_ROOT"

echo "== Syncing R3.1 clean checkpoint runs =="
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
