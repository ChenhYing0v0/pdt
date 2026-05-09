#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:?set REMOTE_HOST before syncing}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-/home/yingch/projects/R_2026_PDT}"
REMOTE_ARTIFACT_ROOT="${REMOTE_ARTIFACT_ROOT:-$REMOTE_REPO_ROOT/artifacts/revision/r3_1_noise_robustness_extension}"
LOCAL_ARTIFACT_ROOT="${LOCAL_ARTIFACT_ROOT:-$ROOT/artifacts/revision/r3_1_noise_robustness_extension}"
IMPORT_NAME="${IMPORT_NAME:-}"

usage() {
  cat >&2 <<'EOF'
usage: sync_r3_1_corruption_extension.sh

Syncs the Weather/ETTm2 R3.1 corruption-evaluation artifacts generated inside
the remote repository checkout. Set REMOTE_HOST and optionally REMOTE_REPO_ROOT.

If a run accidentally wrote artifacts outside the repository checkout, set
REMOTE_ARTIFACT_ROOT to that output directory. Set IMPORT_NAME to keep summary
CSV/TEX files from that alternate root under imports/IMPORT_NAME while merging
eval_runs/ into the standard local artifact tree.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

remote_check_root="${REMOTE_ARTIFACT_ROOT/#\~/\$HOME}"
if ! ssh "$REMOTE_HOST" "test -d ${remote_check_root}"; then
  echo "Remote artifact root does not exist: ${REMOTE_ARTIFACT_ROOT}" >&2
  exit 1
fi

mkdir -p "$LOCAL_ARTIFACT_ROOT"
rsync_includes=(
  --include='/eval_runs/***'
  --include='/figures/***'
)
if [[ -z "$IMPORT_NAME" ]]; then
  rsync_includes+=(
    --include='/robustness_runs.csv'
    --include='/robustness_summary.csv'
    --include='/robustness_table.tex'
  )
fi
rsync -av "${rsync_includes[@]}" --exclude='*' "${REMOTE_HOST}:${REMOTE_ARTIFACT_ROOT}/" "$LOCAL_ARTIFACT_ROOT/"

if [[ -n "$IMPORT_NAME" ]]; then
  import_root="$LOCAL_ARTIFACT_ROOT/imports/$IMPORT_NAME"
  mkdir -p "$import_root"
  for name in robustness_runs.csv robustness_summary.csv robustness_table.tex; do
    remote_file="${REMOTE_ARTIFACT_ROOT%/}/$name"
    if ssh "$REMOTE_HOST" "test -f ${remote_file/#\~/\$HOME}"; then
      rsync -av "${REMOTE_HOST}:${remote_file}" "$import_root/$name"
    fi
  done
  echo "Imported summary files to: $import_root"
fi

echo "Synced corruption artifacts to: $LOCAL_ARTIFACT_ROOT"
