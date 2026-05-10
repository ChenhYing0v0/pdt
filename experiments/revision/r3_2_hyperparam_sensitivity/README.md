# R3.2 / R1.4 Hyperparameter Sensitivity Manifests

This directory contains the standard reviewer-response sensitivity package for:

- `Top-K`: `K = {8, 16, 32, 64, 96}`
- `MCD mask threshold`: `tau = {0.05, 0.10, 0.15, 0.25}`
- datasets: `ETTh1`, `Weather`
- horizon: `pred_len = 336`

The package has 16 manifests in total. Each dataset includes five Top-K runs and
three non-default threshold runs; the default threshold is covered by the
default Top-K run.

Run all manifests on the remote host:

```bash
export DATA_ROOT=/path/to/datasets
scripts/remote/run_r3_2_hyperparam_sensitivity.sh --gpu 0 --skip-predictions
```

Run each dataset on a separate GPU:

```bash
export DATA_ROOT=/path/to/datasets
scripts/remote/run_r3_2_hyperparam_sensitivity.sh --dataset etth1 --pred-len 336 --gpu 0 --skip-predictions
scripts/remote/run_r3_2_hyperparam_sensitivity.sh --dataset weather --pred-len 336 --gpu 1 --skip-predictions
```

Dry-run command expansion:

```bash
scripts/remote/run_r3_2_hyperparam_sensitivity.sh --dry-run --skip-predictions
```

The protocol run directory receives `metrics.json`, `channel_mask.npy`, and
`mask_stats.json` when the model exports a channel mask.
