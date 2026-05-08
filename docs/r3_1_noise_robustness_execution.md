---
title: R3.1 noise robustness execution plan
project: R_2026_PDT
date: 2026-05-08
language: zh-CN
status: ready-for-remote-training
---

# R3.1 noise robustness execution plan

本文档说明 Reviewer #3(1) noise/anomaly robustness 实验的 clean checkpoint
训练落地方式。当前阶段只落实 clean checkpoint 训练与回传保存；test-time
spike/segment corruption evaluation 将在 clean checkpoints 回传后继续执行。

## 1. 目标边界

- 所有 clean checkpoints 重新训练，固定 `seed=2023`。
- 首轮只覆盖 PDT、iTransformer、DLinear。
- 数据集范围暂定为 ETTh1 全 horizon：`pred_len={96,192,336,720}`。
- ECL 暂不进入首轮 clean checkpoint manifests，因为训练耗时较长，且单 `pred_len`
  结果不利于在论文中形成清晰、可展示的 robustness table。
- PDT-CD 暂不进入首轮 clean checkpoint manifests；若 ETTh1 结果不足，再单独恢复
  PDT-CD 或 high-dimensional variant 作为机制补充。

## 2. 代码入口

### PDT

- Runner: `protocol/runners/pdt.py`
- Entry: `baselines/PDT/run.py`
- Model: `PDT`
- Clean checkpoint 输出：
  - nested checkpoint: `checkpoints/{setting}/checkpoint.pth`
  - top-level copy: `best.ckpt`

### iTransformer

- Runner: `protocol/runners/itransformer.py`
- Entry: `baselines/itransformer/run.py`
- Model: `iTransformer`
- Clean checkpoint 输出：
  - nested checkpoint: `checkpoints/{setting}/checkpoint.pth`
  - top-level copy: `best.ckpt`

### DLinear

- Runner: `protocol/runners/dlinear.py`
- Entry: `baselines/PDT/run.py`
- Model: `DLinear`

DLinear 使用 `baselines/PDT` 中现有 model zoo 源码。为使其能被 protocol runner
稳定调用，本次仅做两处最小接入：

1. 在 `baselines/PDT/exp/exp_basic.py` 注册 `DLinear` 到 `model_dict`。
2. 新增 `protocol/runners/dlinear.py`，固定 DLinear 通过
   `baselines/PDT/run.py --model DLinear` 启动。

该路径避免新增第三方源码树，也让 DLinear 与 PDT 使用同一 data split、metric
collector 和 output directory contract。

## 3. Manifests

Clean checkpoint manifests 位于：

```text
experiments/revision/r3_1_noise_robustness/clean_checkpoints/
```

| Manifest | Scope |
|---|---|
| `pdt_etth1_clean.json` | PDT, ETTh1, `pred_len={96,192,336,720}` |
| `itransformer_etth1_clean.json` | iTransformer, ETTh1, `pred_len={96,192,336,720}` |
| `dlinear_etth1_clean.json` | DLinear, ETTh1, `pred_len={96,192,336,720}` |

## 4. Remote training

在远程训练机的仓库根目录执行：

```bash
CONDA_ENV_NAME=pdt OUTPUT_ROOT="$HOME/exp_outputs/r-2026-pdt" \
  bash scripts/remote/run_r3_1_clean_checkpoints.sh --gpu 0
```

后台运行：

```bash
nohup bash scripts/remote/run_r3_1_clean_checkpoints.sh --gpu 0 \
  > r3_1_clean_checkpoints.log 2>&1 &
```

`run_manifest.py` 会将 multi-horizon manifest 展开成单 horizon run directory。
本轮预期 clean run ids 为：

```text
r3_1_clean_pdt_etth1_s2023_pl96
r3_1_clean_pdt_etth1_s2023_pl192
r3_1_clean_pdt_etth1_s2023_pl336
r3_1_clean_pdt_etth1_s2023_pl720
r3_1_clean_itransformer_etth1_s2023_pl96
r3_1_clean_itransformer_etth1_s2023_pl192
r3_1_clean_itransformer_etth1_s2023_pl336
r3_1_clean_itransformer_etth1_s2023_pl720
r3_1_clean_dlinear_etth1_s2023_pl96
r3_1_clean_dlinear_etth1_s2023_pl192
r3_1_clean_dlinear_etth1_s2023_pl336
r3_1_clean_dlinear_etth1_s2023_pl720
```

## 5. Local sync and archive

远程训练完成后，在本地仓库根目录执行：

```bash
REMOTE_HOST=<your-host> REMOTE_RESULTS_ROOT='~/exp_outputs/r-2026-pdt' \
  bash scripts/remote/sync_r3_1_clean_checkpoints.sh
```

该脚本不使用 `--lite`，因此会保留 `checkpoints/`。同步后会额外归档每个 run 的关键文件：

```text
artifacts/revision/r3_1_noise_robustness/clean_checkpoints/{run_id}/best.ckpt
artifacts/revision/r3_1_noise_robustness/clean_checkpoints/{run_id}/metrics.json
artifacts/revision/r3_1_noise_robustness/clean_checkpoints/{run_id}/config.snapshot.json
artifacts/revision/r3_1_noise_robustness/clean_checkpoints/{run_id}/git_meta.json
artifacts/revision/r3_1_noise_robustness/clean_checkpoints/{run_id}/train.log
```

索引文件：

```text
artifacts/revision/r3_1_noise_robustness/clean_checkpoints/index.tsv
```

## 6. Verification performed

本地已完成：

- `py_compile` for touched Python files.
- JSON parse check for all clean checkpoint manifests.
- `bash -n` for remote launch/sync scripts.
- `run_manifest --dry-run` for PDT, iTransformer, and DLinear manifests.

本地未完成：

- Python runtime import / forward / training smoke，因为本机默认 Python 环境无
  `torch`。该检查需在远程 `CONDA_ENV_NAME=pdt` 环境完成。
