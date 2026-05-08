---
title: R3.1 clean checkpoint manifests
project: R_2026_PDT
date: 2026-05-08
language: zh-CN
---

# R3.1 clean checkpoint manifests

本目录保存 Reviewer #3(1) noise/anomaly robustness 实验的 clean checkpoint
训练 manifests。所有 manifests 固定 `seed=2023`，clean checkpoint 需要重新训练，
再作为后续 test-time spike/segment corruption evaluation 的输入。

当前执行范围只包含 ETTh1 全 horizon。ECL 暂不做：训练成本较高，且单
`pred_len` 结果不利于在论文中形成清晰、可展示的 robustness table。

## Manifest scope

| Manifest | Run id base | Produced runs |
|---|---|---|
| `pdt_etth1_clean.json` | `r3_1_clean_pdt_etth1_s2023` | `*_pl96`, `*_pl192`, `*_pl336`, `*_pl720` |
| `itransformer_etth1_clean.json` | `r3_1_clean_itransformer_etth1_s2023` | `*_pl96`, `*_pl192`, `*_pl336`, `*_pl720` |
| `dlinear_etth1_clean.json` | `r3_1_clean_dlinear_etth1_s2023` | `*_pl96`, `*_pl192`, `*_pl336`, `*_pl720` |

## Remote training

在远程训练机的仓库根目录执行：

```bash
CONDA_ENV_NAME=pdt OUTPUT_ROOT="$HOME/exp_outputs/r-2026-pdt" \
  bash scripts/remote/run_r3_1_clean_checkpoints.sh --gpu 0
```

当前 PDT 已完成，`run_r3_1_clean_checkpoints.sh` 只会继续运行 iTransformer 与 DLinear。
如需用两张卡并行运行：

```bash
CONDA_ENV_NAME=pdt OUTPUT_ROOT="$HOME/exp_outputs/r-2026-pdt" \
  nohup bash scripts/remote/run_r3_1_clean_itransformer.sh --gpu 0 \
  > r3_1_clean_itransformer.log 2>&1 &

CONDA_ENV_NAME=pdt OUTPUT_ROOT="$HOME/exp_outputs/r-2026-pdt" \
  nohup bash scripts/remote/run_r3_1_clean_dlinear.sh --gpu 1 \
  > r3_1_clean_dlinear.log 2>&1 &
```

如需顺序后台运行：

```bash
nohup bash scripts/remote/run_r3_1_clean_checkpoints.sh --gpu 0 \
  > r3_1_clean_checkpoints.log 2>&1 &
```

## Local sync and checkpoint archive

远程训练完成后，在本地仓库根目录执行：

```bash
REMOTE_HOST=<your-host> REMOTE_RESULTS_ROOT='~/exp_outputs/r-2026-pdt' \
  bash scripts/remote/sync_r3_1_clean_checkpoints.sh
```

该脚本会用非 `--lite` 模式回传 run directory，保留 `checkpoints/`，并把每个 run 的
`best.ckpt`、`metrics.json`、`config.snapshot.json`、`git_meta.json` 和 `train.log`
归档到：

```text
artifacts/revision/r3_1_noise_robustness/clean_checkpoints/
```

## Corruption evaluation

clean checkpoints 全部回传后，运行 test-time spike/segment corruption evaluation：

```bash
CONDA_ENV_NAME=pdt DATA_ROOT=/home/yingch/projects/R_2026_PDT/baselines/PDT/dataset \
  bash scripts/remote/run_r3_1_corruption_eval.sh --gpu 0 --only-missing
```

若在远程训练机上直接复用 `$OUTPUT_ROOT` 中的 clean runs：

```bash
CONDA_ENV_NAME=pdt OUTPUT_ROOT="$HOME/exp_outputs/r-2026-pdt" \
DATA_ROOT=/home/yingch/projects/R_2026_PDT/baselines/PDT/dataset \
  bash scripts/remote/run_r3_1_corruption_eval.sh --gpu 0 --only-missing \
    --clean-index /tmp/missing-index.tsv \
    --clean-runs-root "$HOME/exp_outputs/r-2026-pdt"
```

汇总输出：

```text
artifacts/revision/r3_1_noise_robustness/robustness_runs.csv
artifacts/revision/r3_1_noise_robustness/robustness_summary.csv
artifacts/revision/r3_1_noise_robustness/robustness_table.tex
```
