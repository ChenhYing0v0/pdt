# GNN Baseline Main Results Extraction

Date: 2026-05-11

This note records the extracted main-result values for the Phase 5 GNN baseline appendix task. The machine-readable values are saved in `gnn_baseline_main_results.csv`.

## Sources

- ForecastGrapher: [arXiv:2405.18036](https://arxiv.org/abs/2405.18036)
- TimeFilter: [arXiv:2501.13041](https://arxiv.org/abs/2501.13041), [OpenReview](https://openreview.net/forum?id=490VcNtjh7)

## Extraction Rules

- Metrics are MSE / MAE.
- ForecastGrapher values are copied from the official detailed result table. The paper table provides per-horizon values; dataset Avg rows in the CSV are locally computed arithmetic means over horizons 96, 192, 336, and 720, rounded to 3 decimals.
- TimeFilter values are copied from the official fixed-lookback $L=96$ detailed table. Its Avg rows are copied from the paper table.
- TimeFilter Traffic is included from the paper result table.
- TimeFilter does not report Exchange in the inspected long-term forecasting table; Exchange is therefore left for rerun from the official code.
- `ECL` in the CSV corresponds to the papers' `Electricity` dataset label.

## Values Ready For Main Table

| Model | Dataset coverage in CSV | Missing dataset action | Avg status |
|---|---|---|---|
| ForecastGrapher | ETTm1, ETTm2, ETTh1, ETTh2, ECL, Exchange, Traffic, Weather | None | Computed from extracted horizons |
| TimeFilter | ETTm1, ETTm2, ETTh1, ETTh2, ECL, Traffic, Weather | Exchange rerun from official code | Copied from paper Avg rows |

## Manual Check Summary

- ForecastGrapher was checked against the LaTeX source table headed by `Ours`; all extracted horizon values are the first MSE / MAE pair in each row.
- TimeFilter was checked against the LaTeX source table headed by `\NAME (Ours)` under the fixed $L=96$ setting; all extracted horizon and Avg values are the first MSE / MAE pair in each row.
- The CSV separates `extracted` and `computed_from_4_horizons` values to avoid confusing paper-reported values with derived averages.
