#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:?set REMOTE_HOST before syncing}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-/home/yingch/projects/R_2026_PDT}"
REMOTE_ARTIFACT_ROOT="${REMOTE_ARTIFACT_ROOT:-$REMOTE_REPO_ROOT/artifacts/revision/r3_1_noise_robustness_extension}"
LOCAL_ARTIFACT_ROOT="${LOCAL_ARTIFACT_ROOT:-$ROOT/artifacts/revision/r3_1_noise_robustness_extension}"

usage() {
  cat >&2 <<'EOF'
usage: sync_r3_1_corruption_extension.sh

Syncs the Weather/ETTm2 R3.1 corruption-evaluation artifacts generated inside
the remote repository checkout. Set REMOTE_HOST and optionally REMOTE_REPO_ROOT.
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
rsync -av "${REMOTE_HOST}:${REMOTE_ARTIFACT_ROOT}/" "$LOCAL_ARTIFACT_ROOT/"
echo "Synced corruption artifacts to: $LOCAL_ARTIFACT_ROOT"
