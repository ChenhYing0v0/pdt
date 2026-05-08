# R_2026_PDT

This repository contains the PDT time-series forecasting implementation and the paper workspace for the KBS revision stage.

## Repository Layout

- `baselines/PDT/`: canonical PDT runtime, reconstructed from the verified old-clone path.
- `protocol/`: thin manifest runner, manifest schema, metric utilities, and artifact recording.
- `experiments/stage1/pdt/`: PDT stage-1 manifests, including smoke and multi-horizon runs.
- `scripts/remote/`: remote launch and result synchronization helpers.
- `manuscript/`: paper and revision materials.

## PDT Manifest Dry-Run

All PDT experiments must use the fixed manifest route:

```text
scripts/remote/run_manifest.sh
  -> python -m protocol.runners.run_manifest
  -> protocol/runners/pdt.py
  -> baselines/PDT/run.py
```

`baselines/PDT_old/` is a provenance snapshot only. Do not use it as the runtime entry for new PDT experiments.

Run manifest validation from the repository root:

```bash
python -m protocol.runners.run_manifest \
  --manifest experiments/stage1/pdt/etth1_smoke.json \
  --dry-run
```

For remote execution, set `DATA_ROOT` to the dataset root that contains `ETT-small/`, `weather/`, `electricity/`, and `traffic/` as needed.

```bash
DATA_ROOT=/path/to/datasets \
CONDA_ENV_NAME=pdt \
scripts/remote/run_manifest.sh experiments/stage1/pdt/etth1_smoke.json --gpu 0 --dry-run
```

The remote wrapper writes results to `~/exp_outputs/r-2026-pdt` by default. Override with `OUTPUT_ROOT` when needed.
