# GNN Baseline Main Results Extraction

Date: 2026-05-12

This note records the extracted main-result values for the Phase 5 GNN baseline appendix task. The machine-readable values are saved in `gnn_baseline_main_results.csv`.

## Sources

- ForecastGrapher: [arXiv:2405.18036](https://arxiv.org/abs/2405.18036)
- TimeFilter: [arXiv:2501.13041](https://arxiv.org/abs/2501.13041), [OpenReview](https://openreview.net/forum?id=490VcNtjh7)

## Extraction Rules

- Metrics are MSE / MAE.
- ForecastGrapher values are copied from the official detailed result table. The paper table provides per-horizon values; dataset Avg rows in the CSV are locally computed arithmetic means over horizons 96, 192, 336, and 720, rounded to 3 decimals.
- TimeFilter values are taken from the reproduced results in `artifacts/summary.csv`, stage `revision_r3_4_gnn_baseline_verify`, using `seed=2023`.
- TimeFilter dataset Avg rows in the CSV are locally computed arithmetic means over horizons 96, 192, 336, and 720 from the reproduced `summary.csv` values, rounded to 3 decimals.
- `ECL` in the CSV corresponds to the papers' `Electricity` dataset label.

## Values Ready For Main Table

| Model | Dataset coverage in CSV | Missing dataset action | Avg status |
|---|---|---|---|
| ForecastGrapher | ETTm1, ETTm2, ETTh1, ETTh2, ECL, Exchange, Traffic, Weather | None | Computed from extracted horizons |
| TimeFilter | ETTm1, ETTm2, ETTh1, ETTh2, ECL, Exchange, Traffic, Weather | None | Computed from reproduced `summary.csv` horizons |

## TimeFilter Reproduced Dataset Averages

| Dataset | Avg MSE | Avg MAE | source |
|---|---:|---:|---|
| ETTm1 | 0.378 | 0.394 | `artifacts/summary.csv` |
| ETTm2 | 0.276 | 0.323 | `artifacts/summary.csv` |
| ETTh1 | 0.434 | 0.437 | `artifacts/summary.csv` |
| ETTh2 | 0.380 | 0.409 | `artifacts/summary.csv` |
| ECL | 0.162 | 0.264 | `artifacts/summary.csv` |
| Weather | 0.245 | 0.273 | `artifacts/summary.csv` |
| Exchange | 0.344 | 0.397 | `artifacts/summary.csv` |
| Traffic | 0.408 | 0.269 | `artifacts/summary.csv` |

## Manual Check Summary

- ForecastGrapher was checked against the LaTeX source table headed by `Ours`; all extracted horizon values are the first MSE / MAE pair in each row.
- TimeFilter rows were replaced with `artifacts/summary.csv` rows whose `baseline` is `timefilter` and whose `stage` is `revision_r3_4_gnn_baseline_verify`.
- The replacement covers 32 reproduced TimeFilter horizon rows and 8 locally computed Avg rows.
- The CSV separates `extracted`, `rerun_result`, `computed_from_4_horizons`, and `computed_from_summary_rerun` values to avoid confusing paper-reported values with derived averages.
