#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_ID="${GPU:-}"
DRY_RUN=0
RUN_ID="${RUN_ID:-r3_1_clean_pdt_weather_canonical_s2023}"
MANIFEST="experiments/revision/r3_1_noise_robustness/clean_checkpoints_extension/pdt_weather_clean_canonical.json"

usage() {
  cat >&2 <<'EOF'
usage: run_r3_1_clean_weather_canonical.sh [--gpu ID] [--dry-run]

Runs Weather clean-checkpoint training through the same canonical PDT protocol
path used by the ETTm2 manifest. Activate the intended Python environment and
set DATA_ROOT before running this script. OUTPUT_ROOT defaults to
$HOME/exp_outputs/r-2026-pdt through scripts/remote/run_manifest.sh.
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
    --dry-run)
      DRY_RUN=1
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

cmd=(bash "$ROOT/scripts/remote/run_manifest.sh" "$MANIFEST" "$RUN_ID" --skip-predictions)
if [[ -n "$GPU_ID" ]]; then
  cmd+=(--gpu "$GPU_ID")
fi
if [[ "$DRY_RUN" -eq 1 ]]; then
  cmd+=(--dry-run)
fi

cd "$ROOT"
"${cmd[@]}"
