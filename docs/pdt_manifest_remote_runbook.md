# PDT Manifest Remote Runbook

## 目标

本说明记录 PDT baseline 通过 protocol manifest 在远程服务器启动的最小闭环。当前验证边界是 syntax、JSON parse、manifest dry-run 和 shell wrapper dry-run；真实训练成功需要在远程服务器具备 conda 环境、CUDA、数据集和 PDT 矩阵文件后确认。

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

命令以 `python -u baselines/PDT/run.py ...` 形式执行，环境由 `protocol/runners/common.py` 构造，并把 repo root 加入 `PYTHONPATH`。

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

本地 dry-run 只能证明 manifest 展开、CLI 参数转发、输出目录快照和 wrapper 拼接是可执行路径；它不会证明数据可读、CUDA 可用、训练收敛或 KBS 返修实验结果已经复现。
