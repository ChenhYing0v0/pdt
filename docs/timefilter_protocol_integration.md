# TimeFilter Protocol Integration

Date: 2026-05-11

## Scope

This document records the local adaptation of the official TimeFilter repository for the Phase 5 GNN baseline appendix task. The imported code is under `baselines/TimeFilter/`, with upstream provenance saved in `baselines/TimeFilter/UPSTREAM_INFO.json`.

## Upstream

- Repository: `https://github.com/TROUBADOUR000/TimeFilter.git`
- Imported commit: `dffde87e4fff0fdeeebbacde03dc1e432e15b3a1`

The local copy keeps the original training flow and model code. The adaptation only adds protocol-facing arguments, seed application, artifact export, a NumPy 2.0 compatibility replacement of deprecated `np.Inf` with `np.inf`, and `drop_last=True` in `data_provider/data_factory.py` for fair comparison with the PDT main-table protocol.

## Runner Flow

The manifest route is `protocol/runners/timefilter.py`.

1. `protocol.runners.run_manifest` loads and expands the manifest.
2. `timefilter.build_command` validates that `entry == baselines/TimeFilter/run.py`.
3. The runner injects protocol fields:
   - `pred_len` from the expanded manifest.
   - `seed` from the manifest.
   - `run_id`, `output_dir`, `metric_policy`, and `selection_policy`.
   - `checkpoints` under the run directory.
4. The command runs `python -u run.py ...` from `baselines/TimeFilter/`.
5. TimeFilter writes protocol artifacts into the run directory after test evaluation.

## Output Artifacts

When `--output_dir` is provided, `baselines/TimeFilter/exp/exp_long_term_forecasting.py` writes:

- `metrics.json`: named scalar metrics `mae`, `mse`, `rmse`, `mape`, `mspe`.
- `metrics.npy`: numeric array `[mae, mse, rmse, mape, mspe]`.
- `result_long_term_forecast.txt`: the original text log, redirected to the run directory.
- `best.ckpt`: copied from the best validation checkpoint if the checkpoint exists.

Large prediction arrays remain disabled, matching the current repo preference for lightweight remote artifacts.

## Exchange Manifest

The Exchange rerun manifest is:

`experiments/revision/r3_4_gnn_baselines/timefilter_exchange_multi_pred_len.json`

TimeFilter reports Traffic in the paper table, but does not report Exchange in the inspected long-term forecasting table. The Exchange rerun therefore reuses the official TimeFilter fixed-lookback ETTh1 script structure and adapts only the dataset route and channel count:

- `seq_len = 96`, `label_len = 48`
- `pred_len = [96, 192, 336, 720]`
- `data = custom`, `data_path = exchange_rate.csv`
- `enc_in = dec_in = c_out = 8`
- `e_layers = 2`, `d_layers = 1`, `factor = 3`
- `patch_len = 2`
- `d_model = 128`, `d_ff = [256, 256, 256, 128]`
- `dropout = 0.8`, `pos = 0`
- `learning_rate = 0.0001`
- `batch_size = 32`, `train_epochs = 10`

Launch wrapper:

```bash
DATA_ROOT=/path/to/dataset/root bash scripts/remote/run_r3_4_timefilter_exchange.sh --gpu 0
```

`DATA_ROOT` must contain `exchange_rate/exchange_rate.csv`. The repo-local dataset layout is compatible with `DATA_ROOT=baselines/PDT/dataset`.

## Full Verification Manifests

To verify TimeFilter results under the same seed and `drop_last=True`, the following manifests are available:

- `experiments/revision/r3_4_gnn_baselines/timefilter_ettm1_multi_pred_len.json`
- `experiments/revision/r3_4_gnn_baselines/timefilter_ettm2_multi_pred_len.json`
- `experiments/revision/r3_4_gnn_baselines/timefilter_etth1_multi_pred_len.json`
- `experiments/revision/r3_4_gnn_baselines/timefilter_etth2_multi_pred_len.json`
- `experiments/revision/r3_4_gnn_baselines/timefilter_ecl_multi_pred_len.json`
- `experiments/revision/r3_4_gnn_baselines/timefilter_weather_multi_pred_len.json`
- `experiments/revision/r3_4_gnn_baselines/timefilter_exchange_multi_pred_len.json`
- `experiments/revision/r3_4_gnn_baselines/timefilter_traffic_multi_pred_len.json`

All manifests use `seed = 2023`, `seq_len = 96`, and `pred_len = [96, 192, 336, 720]`. The ETT/ECL/Weather/Traffic settings follow the official TimeFilter fixed-lookback scripts; Exchange uses the ETTh1-style setting introduced for the missing paper result.

Run all verification manifests on the remote machine:

```bash
DATA_ROOT=/path/to/dataset/root bash scripts/remote/run_r3_4_timefilter_verify_all.sh --gpu 0
```

For a command-only check:

```bash
DATA_ROOT=/path/to/dataset/root bash scripts/remote/run_r3_4_timefilter_verify_all.sh --gpu 0 --dry-run
```

## Verification Boundary

The integration verifies command construction and artifact plumbing. The returned Exchange rerun has collected `metrics.json` files for all four horizons.

## Returned Exchange Results

| pred_len | MSE | MAE | run directory |
|---:|---:|---:|---|
| 96 | 0.081 | 0.200 | `artifacts/runs/r3_4_timefilter_exchange_s2023_pl96` |
| 192 | 0.170 | 0.295 | `artifacts/runs/r3_4_timefilter_exchange_s2023_pl192` |
| 336 | 0.335 | 0.419 | `artifacts/runs/r3_4_timefilter_exchange_s2023_pl336` |
| 720 | 0.643 | 0.605 | `artifacts/runs/r3_4_timefilter_exchange_s2023_pl720` |
| Avg | 0.307 | 0.380 | arithmetic mean over four horizons |

The same values have been added to `manuscript/revision/PDT_revision_v01/checks/gnn_baseline_main_results.csv`.
