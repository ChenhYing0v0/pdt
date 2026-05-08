# PDT old/R2Linear Reproduction Alignment

本文记录 2026-05-08 对当前 `baselines/PDT` 与 old R2Linear 复现实验入口的对齐边界。第二轮排查中，真实 old repository 已被放入 `baselines/PDT_old`，其分支为 `EU-mask-local-20260508`。

## 1. 数据加载与 `drop_last`

已确认 old 与当前 PDT 在 long-term forecasting 的 train/val/test 数据加载上保持一致：

- `old/data_factory.py`：train/val 使用 `shuffle=True`、`drop_last=True`、`batch_size=args.batch_size`；test 使用 `shuffle=False`、`drop_last=True`、`batch_size=1`。
- `baselines/PDT/data_provider/data_factory.py`：train/val 由 `flag != "test"` 得到 `shuffle=True`，test 得到 `shuffle=False`；test 使用 `batch_size=1`，并统一传入 `drop_last=True`。

因此当前 observed gap 不应优先归因于 test 阶段 `drop_last` 不一致。

## 2. 标准参数对齐范围

本轮将 `old/R2Linear_*.sh`、`baselines/PDT/scripts/long_term_forecast/*/*.sh` 和 `experiments/stage1/pdt/*.json` 对齐到 old finetuning 脚本中的标准参数。除 `k_top`、`alpha_init`、`mask_threshold` 外，其余参数固定为对应数据集 old 脚本的标准配置，包括：

- `seq_len=96`，`label_len=48`，`features=M`，`itr=1`，`factor=3`，`d_layers=1`。
- `seed/fix_seed=2023`，`num_workers=8`，`Q_chan_indep=0`，`freeze_R=0`，`CKA_flag=0`。
- `r_rank=96`，`embed_size=16`，`loss_mode=L1`，`rec_lambda=1`，`auxi_lambda=0`。
- 数据集级别的 `learning_rate`、`e_layers`、`d_model`、`d_ff`、`dropout`、`batch_size`、`train_epochs`、`patience`、`lradj` 按 old 脚本固定。

`k_top`、`alpha_init`、`mask_threshold` 已按截图表格写入各数据集与 `pred_len`。截图中 Traffic 三列为空，本轮未臆造 Traffic 的三参表；Traffic 仅保留当前已有的 `pred_len=96`、`k_top=16`、`alpha_init=0.1`、`mask_threshold=0.15`，并同步其余标准参数。

## 3. `LinearEncoder` 对齐

当前 `LinearEncoder.forward` 已恢复 old 版的实际计算顺序：

1. 输入 `x` 形状为 `[B, N, d_model]`。
2. `values = self.v_proj(x)`，形状保持 `[B, N, d_model]`。
3. `attn_base = F.softplus(self.weight_mat)`，形状为 `[1, N, N]`。
4. 若存在 `attn_mask`，将 `[B, 1, N, N]` squeeze 为 `[B, N, N]`，并计算 `attn_masked = attn_base * attn_mask`；否则保持 `[1, N, N]`。
5. `attn = F.normalize(attn_masked, p=1, dim=-1)` 后再执行 `self.dropout(attn)`。
6. `new_x = attn @ values`，随后进入 `out_proj`、residual、`norm1`、FFN、`norm2`。
7. 返回 `(output, None)`，与 old 版一致，不再向上游返回 attention matrix。

这一改动修复了此前当前版缺少 attention dropout、FFN 前归一化顺序不同、返回 attention 而非 `None` 的实现漂移。

## 4. Code-Theory Consistency

预期理论：PDT/R2Linear 使用可学习 token mixing matrix，经 softplus 和 row-normalization 后得到非负归一化的 token relation，再由 Mahalanobis mask 稀疏化关系，并通过 `k_top/alpha_init/mask_threshold` 控制 R 矩阵选择和 mask 软阈值。

代码实现：当前实现已经恢复 old 版 `LinearEncoder` 的 token relation dropout、masked relation broadcasting、`attn @ values` 路径和 `(output, None)` 返回语义；训练入口也固定为 old finetuning 的标准参数，仅保留截图表格指定的三参随数据集和 horizon 变化。

仍然只是 proxy 的部分：截图未提供 Traffic 的三参，因此 Traffic 目前不能声称已完整复现截图设定；此外本轮只做入口和实现对齐，没有运行完整训练验证结果是否回到 old 水平。

可证伪证据：若在相同数据、矩阵文件、seed、CUDA 环境下，当前 `baselines/PDT` 与 `old/R2Linear` 的 resolved CLI、数据切分、`LinearEncoder` forward 中间张量形状均一致，但指标仍显著偏离，则需要继续比较 optimizer/scheduler、early stopping checkpoint 选择、RevIN/normalization、metrics artifact 读取路径和 test-time output inverse transform。

## 5. 第二轮重构：以 `PDT_old` 为基准

实验结果显示参数和局部 `LinearEncoder` 对齐后仍存在显著差距。静态 diff 进一步确认：此前当前 `baselines/PDT` 是 protocol 化后的近似实现，不是 `baselines/PDT_old` 的逐文件复刻；差异覆盖 `run.py`、`exp`、`data_provider`、`layers`、`models`、`utils`，其中任何一个清理改写都可能改变训练轨迹。

本轮已按“完全复刻 old 版本”的优先级重构当前 PDT：

- `baselines/PDT/models/PDT.py` 直接复制自 `baselines/PDT_old/models/R2Linear.py`。
- `baselines/PDT/models/R2Linear.py`、`baselines/PDT/layers/Linear_EncDec.py`、`baselines/PDT/data_provider/data_loader.py` 已与 `PDT_old` 对应文件保持内容一致。
- `baselines/PDT/data_provider/`、`baselines/PDT/layers/`、`baselines/PDT/models/`、`baselines/PDT/utils/` 已补回 old 版依赖文件，避免 `Linear_EncDec.py` 和 old model import 路径漂移。
- `baselines/PDT/exp/exp_long_term_forecasting.py` 恢复 old 训练、验证、测试循环和 old `metric_collector`，只额外在 protocol `output_dir` 存在时写出 `metrics.json`。
- `baselines/PDT/run.py` 以 old `run_IN.py` 为主体，仅增加新架构必要适配：`--seed` 作为 `--fix_seed` alias、`--output_dir/run_id/metric_policy/selection_policy/skip_predictions` 参数、相对路径按 `baselines/PDT` 解析、`--model PDT` 入口、以及 optional `setproctitle`。
- `baselines/PDT/exp/exp_basic.py` 只保留 `PDT` 和 `R2Linear` 映射，避免导入无关 old model zoo 影响 PDT 启动。

保留的新架构差异是运行外壳层面的，不进入 PDT forward 或训练目标：

- protocol runner 仍调用 `baselines/PDT/run.py` 并传入 `--output_dir`。
- `run.py` 在 protocol 模式下把 `checkpoints/results/test_results/log_path` 放入 run directory。
- 结果额外保存 `metrics.json`，便于新架构收集指标。

本地验证边界：当前本机默认 Python 环境缺少 `torch`，无法执行 forward 或训练级 A/B；已完成 `py_compile`、所有 PDT manifest dry-run、核心文件与 `PDT_old` 的静态等价检查。
