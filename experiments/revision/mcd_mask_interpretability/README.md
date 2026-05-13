# MCD Mask Interpretability Manifests

This directory contains candidate PDT runs for selecting one clear visualization
inside Section 5.7, `Analysis of Series Representation`.

The goal is not to claim universal mask behavior across datasets. The runs only
provide multiple candidate settings so that the final manuscript can choose the
most readable learned-mask visualization after all remote outputs are available.

## Candidate Scope

- datasets: `ETTh1`, `Weather`, `ECL`
- horizons: `pred_len = {96, 192, 336, 720}`
- expected artifacts per run:
  - `metrics.json`
  - `channel_mask.npy`
  - `mask_stats.json`

`Weather` is expected to be the most readable main-text candidate because it has
a moderate number of channels. `ETTh1` is useful for sanity checking small
channel graphs, and `ECL` is useful for high-dimensional dependency evidence.

## Run

Run all candidate manifests:

```bash
export DATA_ROOT=/path/to/datasets
scripts/remote/run_mcd_mask_interpretability.sh --gpu 0 --skip-predictions
```

Run one dataset or one horizon:

```bash
scripts/remote/run_mcd_mask_interpretability.sh --dataset weather --gpu 0 --skip-predictions
scripts/remote/run_mcd_mask_interpretability.sh --pred-len 336 --gpu 0 --skip-predictions
```

Dry-run command expansion:

```bash
scripts/remote/run_mcd_mask_interpretability.sh --dry-run --skip-predictions
```

## Analyze After Sync

After remote outputs are synced back into `artifacts/runs`, run:

```bash
export DATA_ROOT=/path/to/datasets
python scripts/revision/analyze_mcd_mask_interpretability.py \
  --runs-root artifacts/runs \
  --output-dir artifacts/analysis/mcd_mask_interpretability
```

The analysis writes a summary CSV and candidate heatmaps. The final manuscript
should integrate only the selected result into Section 5.7.
