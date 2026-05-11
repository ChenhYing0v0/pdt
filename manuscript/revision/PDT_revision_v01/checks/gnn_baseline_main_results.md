# GNN Baseline Main Results Extraction

Date: 2026-05-11

This note records the extracted main-result values for the Phase 5 GNN baseline appendix task. The machine-readable values are saved in `gnn_baseline_main_results.csv`.

## Sources

- ForecastGrapher: [arXiv:2405.18036](https://arxiv.org/abs/2405.18036)
- TimeFilter: [arXiv:2501.13041](https://arxiv.org/abs/2501.13041), [OpenReview](https://openreview.net/forum?id=490VcNtjh7)

## Extraction Rules

- Metrics are MSE / MAE.
- ForecastGrapher values are copied from the official detailed result table. The paper table provides per-horizon values; dataset Avg rows in the CSV are locally computed arithmetic means over horizons 96, 192, 336, and 720, rounded to 3 decimals.
- TimeFilter values are copied from the official fixed-lookback $L=96$ detailed table except Exchange.
- TimeFilter Traffic is included from the paper result table.
- TimeFilter Exchange values are taken from the remote protocol rerun of the official code. Its Avg row is locally computed as the arithmetic mean over horizons 96, 192, 336, and 720, rounded to 3 decimals.
- `ECL` in the CSV corresponds to the papers' `Electricity` dataset label.

## Values Ready For Main Table

| Model | Dataset coverage in CSV | Missing dataset action | Avg status |
|---|---|---|---|
| ForecastGrapher | ETTm1, ETTm2, ETTh1, ETTh2, ECL, Exchange, Traffic, Weather | None | Computed from extracted horizons |
| TimeFilter | ETTm1, ETTm2, ETTh1, ETTh2, ECL, Exchange, Traffic, Weather | None | Paper Avg rows, except Exchange computed from remote rerun |

## TimeFilter Exchange Rerun

| pred_len | MSE | MAE | source |
|---:|---:|---:|---|
| 96 | 0.081 | 0.200 | `artifacts/runs/r3_4_timefilter_exchange_s2023_pl96/metrics.json` |
| 192 | 0.170 | 0.295 | `artifacts/runs/r3_4_timefilter_exchange_s2023_pl192/metrics.json` |
| 336 | 0.335 | 0.419 | `artifacts/runs/r3_4_timefilter_exchange_s2023_pl336/metrics.json` |
| 720 | 0.643 | 0.605 | `artifacts/runs/r3_4_timefilter_exchange_s2023_pl720/metrics.json` |
| Avg | 0.307 | 0.380 | computed from the four rerun horizons |

## Manual Check Summary

- ForecastGrapher was checked against the LaTeX source table headed by `Ours`; all extracted horizon values are the first MSE / MAE pair in each row.
- TimeFilter was checked against the LaTeX source table headed by `\NAME (Ours)` under the fixed $L=96$ setting; all extracted horizon and Avg values are the first MSE / MAE pair in each row.
- TimeFilter Exchange `metrics.json` values were cross-checked against each run's `result_long_term_forecast.txt`.
- The CSV separates `extracted`, `rerun_result`, `computed_from_4_horizons`, and `computed_from_remote_rerun` values to avoid confusing paper-reported values with derived averages.
