# PDT Baseline

该目录已经被整理为当前仓库可用的最小 `PDT` baseline，只保留：

- `long_term_forecast` 单任务
- `PDT` 单模型
- 当前训练/测试实际依赖的数据加载、模型、层、工具函数
- 与仓库 `protocol/` 对接所需的 `run.py`

## 目录约定

- `run.py`
  PDT 的唯一 Python 入口，支持本地脚本调用和 `protocol.runners.run_manifest` 调用。
- `models/PDT.py`
  当前实际使用的模型实现，仓库内统一以 `PDT` 作为模型名称。
- `scripts/long_term_forecast/`
  仅保留各数据集的 `PDT` 训练脚本。

## 推荐入口

本地脚本：

```bash
bash baselines/PDT/scripts/long_term_forecast/ETT/PDT_ETTh1.sh
```

protocol dry-run：

```bash
python3 scripts/local/run_manifest.py \
  --manifest experiments/stage1/pdt/etth1_smoke.json \
  --dry-run
```
