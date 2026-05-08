# PDT Manifest Remote Runbook

## 目标

本说明记录 PDT baseline 通过 protocol manifest 在远程服务器启动的最小闭环。当前 PDT canonical route 已由远程实验确认：该路径与之前 old version 实验结果完全一致，后续所有 PDT 相关实验均固定走此路径。

Canonical route:

```text
scripts/remote/run_manifest.sh
  -> python -m protocol.runners.run_manifest
  -> protocol/runners/pdt.py
  -> baselines/PDT/run.py
```

`baselines/PDT/` 是已按 `baselines/PDT_old` old-clone 复刻并验证的运行目录；`baselines/PDT_old/` 只保留为来源快照和静态对照，不作为后续实验入口。

## 功能模块

### Manifest dispatcher

入口是 `protocol/runners/run_manifest.py`。它读取 JSON manifest，应用 `--set` 覆盖项，按 `pred_len` 展开 batch manifest，并把单个 child manifest 交给对应 baseline builder。

当前有效 baseline 注册项为：

- `pdt`: `protocol/runners/pdt.py`
- `itransformer`: `protocol/runners/itransformer.py`

PDT 仓库不再注册 `fredr`，避免缺失 `protocol.runners.fredr` 时阻断所有 manifest。

### PDT builder

`protocol/runners/pdt.py` 将 manifest 中的 `args` 转成 PDT CLI 参数，并补充：

- `pred_len`
- `seed`
- `run_id`
- `output_dir`
- `metric_policy`
- `selection_policy`

命令固定以 `python -u baselines/PDT/run.py ...` 形式执行，环境由 `protocol/runners/common.py` 构造，并把 repo root 加入 `PYTHONPATH`。

`protocol/runners/pdt.py` 会检查 manifest 中的 `entry` 是否等于 `baselines/PDT/run.py`。如果未来 PDT manifest 指向其它入口，runner 会直接报错，避免绕开已验证路径。

### Remote wrapper

`scripts/remote/run_manifest.sh` 是远程启动入口。默认约定：

- `CONDA_ENV_NAME=pdt`
- `OUTPUT_ROOT=$HOME/exp_outputs/r-2026-pdt`
- `--gpu ID` 转成 `--set env.CUDA_VISIBLE_DEVICES=ID`
- 远程后台管理建议由外层 `tmux` session 负责，wrapper 本身只执行前台命令。

### Result sync

`scripts/remote/sync_results.sh` 默认从 `~/exp_outputs/r-2026-pdt` 同步结果到本地 `artifacts/runs/`。`--lite` 会跳过 `*.npz` 和 `checkpoints/`。

## 远程运行前提

1. 远程仓库 checkout 包含 `baselines/PDT/dataset/RRR_mats/` 和 `baselines/PDT/dataset/PCCA_mats/`。
2. `DATA_ROOT` 指向远程数据根目录，并包含 manifest 中的子目录。
3. `CONDA_ENV_NAME` 指向安装了 `torch`、`numpy`、`pandas`、`scikit-learn` 等依赖的环境。
4. GPU 选择通过 `--gpu` 或 `GPU` 环境变量传入。

## 最小命令

```bash
DATA_ROOT=/path/to/datasets \
CONDA_ENV_NAME=pdt \
scripts/remote/run_manifest.sh experiments/stage1/pdt/etth1_smoke.json --gpu 0 --dry-run
```

在 `tmux` session 中正式运行：

```bash
DATA_ROOT=/path/to/datasets \
CONDA_ENV_NAME=pdt \
scripts/remote/run_manifest.sh experiments/stage1/pdt/etth1_smoke.json smoke_etth1 --gpu 0
```

## 验证边界

本地 dry-run 只能证明 manifest 展开、CLI 参数转发、输出目录快照和 wrapper 拼接是可执行路径；它不会证明数据可读、CUDA 可用或训练收敛。

2026-05-08 远程实验已确认 canonical PDT route 可以正常完成实验，并且结果与之前 old version 实验结果完全一致。因此后续 PDT 实验的复现基准不是 `baselines/PDT_old/run_IN.py` 直接启动，而是通过上述 protocol route 调用 `baselines/PDT/run.py`。
