---
title: Phase 5 reviewer comment action matrix
project: R_2026_PDT
manuscript_id: KNOSYS-D-26-03771
manuscript: PDT: Predictive Domain Transform for Multivariate Time Series Forecasting
journal: Knowledge-Based Systems
date: 2026-05-07
language: zh-CN
status: draft-action-matrix
---

# Phase 5 reviewer comments 逐条处理文档

本文档对应 `manuscript/revision/revision_work_plan.md` 中的
`Phase 5：reviewer comments 处理块`。本轮处理目标是把每条 reviewer
comment 拆分为 `required action`、`manuscript change`、`experiment need`
和 `response strategy`，并给出可直接迁移到 response letter 的英文回复草稿。

## 当前稿件理解

### 论文主线

当前稿件提出 `Predictive Domain Transform Method (PDTM)` 与基于 PDTM 的
`PDT` 预测框架。核心逻辑如下：

1. MTSF 的难点被界定为两个方向：长程 temporal correlation 与复杂
   inter-variable dependency。
2. Linear-based models 高效但非线性表达不足；Transformer-based models
   表达强但计算/内存成本高。
3. 频域或 wavelet 等 fixed-basis transforms 可缓解 time-domain 表达瓶颈，
   但固定基不直接针对 forecasting error 优化。
4. PDTM 通过训练集滑窗构造 past-future pair，估计
   $\Sigma_{xx}$ 与 $\Sigma_{xy}$，在 whitened constraint 下求解
   trace maximization，得到预测最优低秩 basis。
5. PDT 利用 PDTM 的 energy concentration：`Linear-Route` 只外推 Top-K
   dominant components，`Encoder-Route` 对 full spectrum 做非线性与
   inter-channel refinement。
6. `MCD` 通过 Top-K P-domain features 计算 weighted distance，再用
   threshold-sigmoid 生成 channel mask，与 static learnable correlation
   matrix 一起进入 lightweight Linear-Attention。
7. 实验部分目前包括主结果、可视化、PDTM/dual-route/MCD ablation、efficiency、
   PDTM generality、architecture preference、look-back sensitivity 和
   representation visualization。

### 已有证据

- 主表用 MSE/MAE 展示 PDT 在 8 个 benchmark datasets 上整体领先。
- Ablation 已覆盖 `w/o PDTM`、`P-Linear`、`P-Encoder`、`PDT-CI`、`PDT-CD`
  和 `PDT-Att`。
- Current code already computes `rmse`, `mape`, `mspe`, `rse`, and `corr`
  through `protocol.metrics.compute_metrics` and writes them to `metrics.json`。
- Current PDT code can save `channel_mask.npy` and `linear_encoder_A.npy` when
  output hooks are enabled, which can support mask interpretability analysis.
- Current manuscript already includes a representation visualization, but it
  visualizes latent correlation matrices rather than the learned MCD mask itself.

### 当前薄弱点

- PDTM 与 FreDF/TransDF 的 conceptual distinction 只有 Related Work 中的一句，
  不足以消除 reviewer 对 data-adaptive transform family 的疑问。
- PDTM 推导已经完整但过于压缩，CCA / reduced-rank regression 的直觉没有独立展开。
- Method notation 中 `T` 同时接近 total sequence notation 和 prediction horizon，
  且 `B/G/W/H` 的 transition 缺少 symbol table。
- Hyperparameters 的经验选择可从 manifests 和 code 中追溯，但 manuscript 没有给出
  selection guideline 或 sensitivity analysis。
- Robustness 的现有证据主要是 visual analysis 中的 qualitative claim，不足以支撑
  "suppress noise" 与 "resilience to outliers"。
- Reviewer #3 要求 graph neural network baselines、relative metrics 和匿名代码链接，
  这些都属于 response letter 中不可回避的 action items。

## 处理总策略

本轮建议采用 `accept + clarify + targeted experiment` 的组合策略。

- Reviewer #1 的 comments 多为 clarity-oriented，不需要新增大规模训练即可处理；
  应通过正文重写、symbol table、comparison paragraph、mask visualization 和
  hyperparameter guideline 解决。
- Reviewer #3 的 comments 多为 evidence-oriented，至少需要新增轻量实验或补充已存在
  metrics；其中 robustness、hyperparameter sensitivity 和 GNN baseline 是高优先级。
- 不建议在 response 中承诺 "proved eigenvalue decay"。应改为证明 Top-K retainable
  predictability energy 的 objective-level relation，并明确 eigenvalue decay 是
  data-dependent empirical observation。
- 不建议在未完成 code release route 前写死 public GitHub URL。Reviewer #3.6 要求
  anonymized repository，应优先准备 anonymized/private-for-review link，或在 response
  中说明 repository is prepared for review with parameter settings.

## Reviewer #1

### Reviewer #1.1 PDTM motivation 与 FreDF/TransDF 区分

**原始意见**

> The motivation of PDTM could be further clarified, especially its distinction
> from existing data-adaptive transforms (e.g., FreDF, TransDF). A more explicit
> comparison at the conceptual level would improve clarity.

**分类与优先级**

- 类型：Minor / clarity
- 优先级：中高
- 策略：Accept + Clarify

**Required action**

补充 conceptual-level comparison，明确 PDTM 与 FreDF/TransDF 的优化目标不同：

- FreDF/TransDF：更偏向 decorrelation / domain transformation，用于缓解
  autocorrelation 或 fixed Fourier domain 的限制。
- PDTM：直接以 future prediction error 为目标，通过 $\Sigma_{xy}$ 与
  low-rank predictor 得到 prediction-optimal basis。
- PDTM 不是单纯 "another adaptive transform"，而是监督式 predictive transform；
  basis 的重要性由 past-to-future cross-covariance 决定。

**Manuscript change**

建议修改两个位置：

1. Introduction 中 fixed-basis transform 后增加 1 段 conceptual motivation。
2. Related Work / Domain Transform 中把当前一句 FreDF/TransDF 对比扩展为小段或
   compact comparison table。

建议新增表格列：

| Method family | Basis/source | Main objective | Forecasting role |
|---|---|---|---|
| Fourier/Wavelet/FITS/FreTS | fixed basis | spectral sparsity/reconstruction | efficient representation |
| FreDF/TransDF | data-adaptive decorrelation basis | decorrelation/autocorrelation mitigation | reduce domain redundancy |
| PDTM | supervised predictive basis from $\Sigma_{xy}$ | minimize forecasting error under rank constraint | concentrate future-relevant information |

**Experiment need**

不需要新增实验。当前 PDT-iTrans vs Freq-iTrans 的 generality table 可作为辅助证据。

**Response strategy**

接受建议，并说明已经扩展 motivation 与 Related Work comparison。

**Response draft**

```text
We thank the reviewer for this valuable suggestion. We agree that the
conceptual distinction between PDTM and existing adaptive domain transforms
should be made more explicit. In the revised manuscript, we have expanded the
motivation in the Introduction and Related Work sections. We now clarify that
FreDF/TransDF mainly target data-adaptive decorrelation or autocorrelation
mitigation, whereas PDTM is derived from a supervised forecasting objective:
it uses the past-future cross-covariance to find a low-rank predictive basis
that directly minimizes the future prediction error. We also added a compact
conceptual comparison table to summarize the difference between fixed-basis,
decorrelation-oriented adaptive transforms, and the proposed predictive-domain
transform.
```

**预计后续工作**

- 写入 manuscript：是。
- 写入 response letter：是。
- 写入 list of changes：是。
- 新实验：否。

### Reviewer #1.2 PDTM derivation intuition

**原始意见**

> The derivation of the predictive domain transform (Section 3.2) is
> mathematically sound, but some intermediate steps are condensed. Providing
> additional intuition (e.g., connection to CCA or low-rank regression) would
> enhance readability for a broader audience.

**分类与优先级**

- 类型：Minor / clarity
- 优先级：高
- 策略：Accept + Clarify

**Required action**

补充推导直觉，而不是重写完整证明。重点是解释：

- Eq. objective 本质是 reduced-rank regression：用 rank-$r$ latent variable
  $z=B^\top x$ 压缩过去窗口，再由 $G$ 预测未来。
- Whitened constraint $B^\top \Sigma_{xx} B=I$ 去掉 scale/rotation ambiguity，
  使每个 latent component 可比较。
- CCA connection：CCA 最大化两个视图投影后的 correlation；PDTM 借用 whitening
  idea，但目标是 future prediction error / squared cross-covariance energy。
- Trace maximization 的 eigenvalues 表示每个 projected direction 对线性预测误差
  reduction 的贡献。

**Manuscript change**

在 Section 3.2 的 closed-form solution 前后增加一个 "Intuition" paragraph 或
remark block。建议措辞：

- "PDTM can be viewed as supervised reduced-rank regression in a whitened input
  space."
- "The CCA connection is methodological rather than identical: CCA searches for
  mutually correlated projections of two views, whereas PDTM searches for input
  directions that best explain future targets under a linear forecasting head."

**Experiment need**

不需要新增实验。

**Response draft**

```text
We thank the reviewer for this helpful comment. We have revised Section 3.2
to provide more intuition behind the derivation. Specifically, we now explain
that PDTM can be interpreted as supervised reduced-rank regression in a
whitened input space: the transform B compresses the historical window into
rank-r predictive variables, and the head G maps these variables to the future
horizon. We also clarify the connection to CCA. PDTM borrows the whitening
constraint used in CCA to remove scale and rotation ambiguity, but its
objective is different: rather than maximizing correlation between two views,
PDTM maximizes the future-predictive covariance energy and therefore directly
minimizes forecasting error under a rank constraint.
```

**预计后续工作**

- 写入 manuscript：是。
- 写入 response letter：是。
- 写入 list of changes：是。
- 新实验：否。

### Reviewer #1.3 Notation consistency 与 symbol table

**原始意见**

> The notation in the methodology section could be improved for consistency,
> particularly the transitions between matrices (e.g., B, G, W) and covariance
> terms. A summary table of symbols would be helpful.

**分类与优先级**

- 类型：Minor / clarity
- 优先级：高
- 策略：Accept

**Required action**

新增 symbol table，并修正 methodology notation 的局部歧义。特别需要处理：

- $X \in R^{N \times T}$ 与 prediction horizon $T$ 容易冲突。
- $\mathbf{x}$ / $\mathbf{y}$ 是单变量或单通道窗口时的 vectorized notation；
  $X$ 是 multivariate series。
- $\mathbf{B}$ 是 original transform basis，$\mathbf{W}$ 是 whitened-space
  orthogonal basis，$\mathbf{H}$ 是 predictive covariance matrix。
- $\mathbf{G}$ 是 latent-to-future linear head。
- $\Sigma_{xx}$ / $\Sigma_{xy}$ / $\Sigma_{yx}$ 的 shape 应明确。

**Manuscript change**

建议在 Problem Definition 后或 Section 3.2 开头新增 `Table: Summary of symbols`。
最小表项：

| Symbol | Shape | Meaning |
|---|---|---|
| $N$ | scalar | number of variates |
| $L$ | scalar | look-back length |
| $H$ or $T_f$ | scalar | prediction horizon length |
| $\mathbf{x}$ | $R^L$ | historical window for PDTM derivation |
| $\mathbf{y}$ | $R^H$ | future target window for PDTM derivation |
| $\mathbf{B}$ | $R^{L \times r}$ | predictive transform basis |
| $\mathbf{G}$ | $R^{r \times H}$ | predictive-domain linear head |
| $\Sigma_{xx}$ | $R^{L \times L}$ | input covariance |
| $\Sigma_{xy}$ | $R^{L \times H}$ | past-future cross-covariance |
| $\mathbf{W}$ | $R^{L \times r}$ | whitened-space orthogonal basis |
| $\mathbf{H}$ | $R^{L \times L}$ | predictive covariance matrix |
| $r$ | scalar | reduced rank of PDTM |
| $K$ | scalar | number of Top-K components used by Linear-Route and CMG |
| $\mathcal{M}$ | $R^{N \times N}$ | MCD channel mask |

如时间允许，建议把 Problem Definition 中 future horizon 的符号从 `T` 改为 `H`，
但这会带来全文公式与表述联动。最小可行方案是保留原符号但在 symbol table 中明确
局部含义。

**Experiment need**

不需要。

**Response draft**

```text
We thank the reviewer for pointing this out. We have added a symbol table in
the methodology section and revised the surrounding text to make the transitions
between B, G, W, H and the covariance matrices explicit. The table now specifies
the shape and meaning of the historical window, future target, predictive
transform basis, prediction head, covariance terms, reduced rank r, Top-K
components, and the MCD mask. We also clarified that W denotes the orthogonal
basis in the whitened input space, while B is the final transform basis in the
original input space.
```

**预计后续工作**

- 写入 manuscript：是。
- 写入 response letter：是。
- 写入 list of changes：是。
- 新实验：否。

### Reviewer #1.4 Hyperparameter choice

**原始意见**

> The choice of hyperparameters, such as the reduced rank r, Top-K components,
> and mask threshold parameters in MCD, is not sufficiently discussed. Including
> sensitivity analysis or guidelines for selecting these parameters would
> strengthen the paper.

**分类与优先级**

- 类型：Major / experiment + clarity
- 优先级：高
- 策略：Accept + Experiment

**Required action**

该意见与 Reviewer #3.2 重复，应合并处理。必须补充：

1. Hyperparameter selection guideline：
   - $r$：不超过 look-back length $L$，默认使用 $r=L=96$；若计算或存储受限，可用
     energy cumulative ratio 或 validation MSE 选择较小 $r$。
   - $K$：依据 cumulative predictability energy 和 validation performance 选择；
     当前 manifests 已体现 per-horizon Top-K，例如 ECL 为 `[96,64,32,32]`，
     ETTh1 为 `[16,96,64,96]`。
   - mask threshold $\tau$：控制 mask sparsity，当前值主要在 `0.05-0.25` 区间。
   - sharpness $k$：控制 sigmoid 硬化程度，当前代码默认 `100.0`。
2. Sensitivity experiment：
   - 最小集合：ETTh1、Weather、ECL 三个 datasets。
   - 最小 horizons：96、336、720；若时间不足，使用 Avg over 4 horizons 的
     validation/test summary。
   - Grid：
     - $K \in \{8,16,32,64,96\}$
     - $\tau \in \{0.05,0.10,0.15,0.25,0.35\}$
     - sharpness $k \in \{50,100,200\}$ 可作为 appendix-only；主文优先展示 $K$ 和
       $\tau$。
   - Output：MSE、MAE、RMSE；同时报告 mask density 作为 interpretability 附助。

**Manuscript change**

- Section 4 / Ablation 后新增 `Hyperparameter Sensitivity` subsection。
- Method 中 MCD 或 Linear-Route 后增加 guideline paragraph。
- Table or figure：
  - `K sensitivity`：x-axis K，y-axis MSE/RMSE，分 dataset 或 horizon。
  - `threshold sensitivity`：x-axis tau，y-axis MSE 和 mask density。

**Experiment need**

需要。当前本地没有结果 artifacts，不能在 response 中声称已完成。应列为必须补做。

**Response draft**

```text
We thank the reviewer for this important suggestion. We agree that the choices
of r, K and the MCD threshold should be better justified. In the revised
manuscript, we added practical selection guidelines and a new hyperparameter
sensitivity analysis. The reduced rank r is selected according to the
look-back length and the cumulative predictive energy, while K is selected by
balancing the retained predictive energy and validation performance. We also
analyze the effect of the MCD threshold, which controls the sparsity of the
learned channel mask. The additional results show that PDT is stable over a
reasonable range of K and threshold values, and that overly small K or overly
sparse masks can reduce the retained predictive information.
```

**预计后续工作**

- 写入 manuscript：是。
- 写入 response letter：是。
- 写入 list of changes：是。
- 新实验：是，优先级高。
- 验证：使用 canonical `experiments/stage1/pdt/*.json` 与
  `scripts/local/run_manifest.py` 生成变体；JSON parse + dry-run + 训练结果汇总。

### Reviewer #1.5 MCD learned mask interpretability

**原始意见**

> Although the Masked Channel Dependent (MCD) strategy is well-motivated, the
> interpretability of the learned mask is not explored. Visualizing or analyzing
> the learned channel relationships could provide additional insights.

**分类与优先级**

- 类型：Major / analysis
- 优先级：高
- 策略：Accept + Analysis

**Required action**

当前 manuscript 的 Fig. representation 展示的是 learned representation
correlation，不是 learned mask。需要补充真实 MCD mask 的可视化与统计：

- Visualize $\mathcal{M}$ on ECL/Weather/ETTh1。
- 对比：
  1. ground-truth future correlation matrix；
  2. learned MCD mask；
  3. linear attention matrix after mask；
  4. PDT-CD without mask 的 learned representation。
- Quantitative interpretability metrics：
  - mask density：$\frac{1}{N^2}\sum_{ij}\mathbb{1}(M_{ij}>\eta)$ 或 soft density。
  - alignment with future correlation：Pearson/Spearman between mask value and
    absolute future-correlation value。
  - top edges 示例：展示 ECL/Weather 中被保留的 high-correlation channel pairs。

**Manuscript change**

- 在 Analysis / Series Representation 后新增 `Analysis of Learned Channel Mask`。
- 新 figure 可命名为 `Figure_9_mask_interpretability.png`。
- 如果版面紧张，主文放 ECL heatmap，appendix/supplement 放更多 dataset。

**Experiment need**

需要轻量分析，不需要重新训练；如果 checkpoint 和 prediction run 可复用，可以用测试阶段
保存 `channel_mask.npy` 与 `linear_encoder_A.npy`。当前 code 已有保存函数，但需要确认
runner 是否启用保存 hook。

**Response draft**

```text
We thank the reviewer for this insightful suggestion. We agree that directly
examining the learned MCD mask can make the channel-dependent mechanism more
interpretable. In the revised manuscript, we added a new analysis of the
learned channel mask. We visualize the MCD mask and compare it with the
future-series correlation structure, and we further report mask density and
correlation-alignment statistics. The results show that MCD learns a sparse
and structured dependency pattern, preserving strongly related variates while
filtering weak or noisy channel interactions.
```

**预计后续工作**

- 写入 manuscript：是。
- 写入 response letter：是。
- 写入 list of changes：是。
- 新实验：轻量分析；优先复用 checkpoint。
- 验证：检查 `channel_mask.npy` / figure file 存在；统计脚本输出可复现 CSV。

## Reviewer #3

### Reviewer #3.1 Robustness and boundary conditions

**原始意见**

> It is recommended that the experimental section include quantitative tests of
> robustness and boundary conditions... employ more rigorous anomaly injection
> tests.

**分类与优先级**

- 类型：Major / experiment
- 优先级：最高
- 策略：Accept + Experiment

**Required action**

必须补充定量 robustness tests。当前 visual analysis 不能支撑 reviewer 要求。建议实验：

1. Test-time anomaly injection：
   - spike noise：随机选择输入窗口中 $p \in \{1\%,5\%,10\%\}$ 的 time-channel
     positions，加入 $a\sigma$，$a \in \{1,2,3\}$。
   - segment corruption：随机短片段置换或放大，用于模拟 sensor glitch。
   - Gaussian noise 可作为辅助，但 reviewer 明确说 occasional noise events，
     spike/segment 更贴切。
2. 评估对象：
   - PDT vs PDT-CD vs PDT-CI 或 PDT vs iTransformer/PatchTST/DLinear。
   - 最小推荐：PDT、PDT-CD、iTransformer、DLinear。
3. Datasets：
   - ECL：high-dimensional channel noise，与 MCD claim 直接相关。
   - Weather：中等维度、非平稳明显。
   - ETTh1：小维度标准 benchmark。
4. Metrics：
   - clean MSE/RMSE；
   - corrupted MSE/RMSE；
   - degradation ratio：$(MSE_{corrupt}-MSE_{clean})/MSE_{clean}$。

**Manuscript change**

- 在 Experiments 或 Analysis 中新增 `Robustness to injected anomalies`。
- 修改 Forecasting Visualization 中的 qualitative statement，避免只靠图说
  "strong resilience"。
- 新增 robustness table：columns 为 clean / spike / segment / degradation ratio。

**Experiment need**

需要。当前 code 的 `--add_noise` 是 frequency-domain perturbation，且作用在
train/validation range，不完全等价于 anomaly injection。建议新增独立 test-time
corruption evaluation script，避免修改训练 protocol。

**Response draft**

```text
We thank the reviewer for this important suggestion. We agree that the visual
case study alone is not sufficient to support the robustness claim. We have
therefore added quantitative anomaly-injection experiments. Specifically, we
evaluate the trained models under test-time spike and segment perturbations and
report both the corrupted-set error and the degradation ratio relative to the
clean test set. The additional results show that PDT has a smaller performance
degradation than the compared baselines, supporting the claim that the MCD
module helps suppress interference from irrelevant or corrupted channels.
```

**预计后续工作**

- 写入 manuscript：是。
- 写入 response letter：是。
- 写入 list of changes：是。
- 新实验：是，优先级最高。
- 验证：新增脚本 `analysis/` 或 `scripts/local/` 后运行 py_compile、dry-run、
  sample corruption sanity check。

### Reviewer #3.2 Key hyperparameter sensitivity

**原始意见**

> It is recommended to include sensitivity analyses for key hyperparameters...

**分类与优先级**

- 类型：Major / experiment
- 优先级：高
- 策略：Accept + Experiment

**处理方式**

与 Reviewer #1.4 合并处理。Response letter 中应分别回复，但说明同一组新增实验同时
address both reviewers' concerns。

**Response draft**

```text
We thank the reviewer for this valuable suggestion. This concern is closely
related to Reviewer #1's comment on hyperparameter selection. We have added a
new sensitivity analysis for the Top-K components and the MCD threshold, together
with practical selection guidelines. The results indicate that PDT remains
stable within a reasonable range of K and threshold values, while extremely
small K or overly sparse masks can reduce the retained predictive information.
These findings have been added to the revised experimental section.
```

**预计后续工作**

- 与 Reviewer #1.4 共用实验和 manuscript section。

### Reviewer #3.3 RMSE or relative error metrics

**原始意见**

> The experiments utilised only two scale-dependent metrics: MSE and MAE...
> recommended to include metrics such as RMSE.

**分类与优先级**

- 类型：Minor-to-Major / metric reporting
- 优先级：高但执行成本低
- 策略：Accept

**Required action**

补充 RMSE。当前 protocol 已经计算 RMSE；如果 `metrics.json` 完整存在，可直接汇总。
如果原始 experiments 只保留 MSE/MAE，可由 MSE 计算 RMSE，但要在内部记录中标明
RMSE is derived from reported MSE。若 response 中说 "we report RMSE"，最好在
revised table/appendix 中明确展示。

关于 "relative error"，reviewer 句子中把 RMSE 作为例子，但 RMSE 仍是 scale-dependent。
更稳妥的处理：

- 主文追加 RMSE。
- Supplementary 或 appendix 追加 RSE/MAPE/MSPE，其中 current code 已有 RSE/Corr。
- Response 中避免声称 RMSE 是 relative metric；可说 "we added RMSE and additional
  normalized/relative metrics where appropriate"。

**Manuscript change**

- 主结果表如版面不允许三指标，建议主文保留 MSE/MAE，appendix 增加 RMSE/RSE。
- 在 Experimental Setup / Metrics paragraph 中说明 reported metrics。

**Experiment need**

通常不需要重新训练；需要汇总 existing metrics 或从 prediction artifacts 重新 evaluate。

**Response draft**

```text
We thank the reviewer for this suggestion. We have expanded the metric
reporting in the revised manuscript. In addition to MSE and MAE, we now report
RMSE and provide normalized/relative metrics in the supplementary results where
appropriate. This makes the cross-dataset comparison more informative while
keeping the main table readable.
```

**预计后续工作**

- 写入 manuscript：是。
- 写入 response letter：是。
- 写入 list of changes：是。
- 新实验：一般否；需要 metrics aggregation。
- 验证：检查 `metrics.json` 中 `rmse` 字段；如通过 MSE 派生，记录公式。

### Reviewer #3.4 Graph neural network baselines

**原始意见**

> The baselines selected for the experiments include mainstream Transformer and
> Linear models, but overlook some recent advanced graph neural network time
> series forecasting models specifically designed to explicitly model the
> topological structure of multivariate variables.

**分类与优先级**

- 类型：Major / baseline comparison
- 优先级：高
- 策略：Accept + Experiment, with scoped justification

**Required action**

该意见需要谨慎处理，因为通用 MTSF benchmark 不一定提供真实 graph topology。
建议策略：

1. 承认 graph-based baselines 有价值。
2. 增加 Related Work 中 graph/topology-based TSF paragraph。
3. 增加一组 graph baseline comparison，但限定 scope：
   - 如果不依赖外部 topology：优先考虑 MTGNN、StemGNN。
   - 如果使用 traffic-specific topology：可在 Traffic 上加入 traffic-flow GNN 或
     hypergraph model，例如已有 `ref.bib` 中的 HR-DHAN。
4. 在 manuscript 中明确：PDT 不假设先验 graph；MCD 学习 soft/sparse channel
   relation。因此与需要 known topology 的交通图模型并非完全同一设定。

**Candidate baselines**

- MTGNN: graph learning + temporal convolution for multivariate time series.
- StemGNN: spectral temporal graph neural network for multivariate forecasting.
- AGCRN/DCRNN/STGCN: strong traffic forecasting baselines,但通常依赖 road-network
  adjacency，跨 ECL/Weather/Exchange 不一定公平。
- HR-DHAN: 当前 `ref.bib` 已有 KBS 2025 traffic-flow paper，可作为 Traffic-specific
  discussion 或 baseline candidate。

**Manuscript change**

- Related Work 增加 graph neural TSF paragraph。
- Experiments / Baselines 中说明是否包含 graph baselines 以及 graph information
  availability。
- 如果运行完成，在 main table 或 supplementary table 中增加 GNN baseline 结果。
- 如果无法在所有 datasets 上公平运行，主文需解释 "we include graph-learning baselines
  that do not require predefined topology; topology-dependent traffic models are discussed
  separately or evaluated only on Traffic."

**Experiment need**

建议至少补一个可跨 dataset 的 graph-learning baseline。最低可接受范围：

- MTGNN + StemGNN on ECL, Weather, Traffic, ETTh1。
- Horizons: 96/192/336/720 若时间允许；否则至少 96/336/720 并解释 computational cost。

**Response draft**

```text
We thank the reviewer for this important suggestion. We agree that graph-based
time-series forecasting models provide a relevant comparison because they
explicitly model inter-variate topology. In the revised manuscript, we have
expanded the related work discussion on graph neural forecasting models and
added graph-based baselines where the benchmark setting allows a fair
comparison. Since several traffic-oriented GNN models require a predefined road
network adjacency that is unavailable for generic MTSF datasets such as ECL,
Weather and Exchange, we focus on graph-learning baselines that can infer
inter-variate relationships from data, and we discuss topology-dependent traffic
models separately. This also clarifies the distinction between PDT, which learns
a sparse channel relation without requiring an external graph, and models that
depend on known graph topology.
```

**预计后续工作**

- 写入 manuscript：是。
- 写入 response letter：是。
- 写入 list of changes：是。
- 新实验：建议做，优先级高。
- 验证：新增 baseline 结果必须记录 config、seed、dataset split 和 artifact path。

### Reviewer #3.5 Top-K predictive information concentration theory boundary

**原始意见**

> The paper mentions that PDTM can concentrate 'predictive information' into
> the Top-K features; however, this is an empirical observation, and there is no
> proof as to why eigenvalue decay necessarily holds, or whether it depends on
> the data distribution.

**分类与优先级**

- 类型：Major / theory clarification
- 优先级：最高
- 策略：Clarify + Partial Defend + Limitation

**Required action**

这里不能强行证明 universal eigenvalue decay。应改成：

- 可证明部分：在 PDTM objective 下，每个 eigenvalue 对应一个 predictive direction
  的 squared cross-covariance / error reduction contribution；保留 Top-K 最大
  eigenvalues 是 rank-K optimum。
- 不可证明部分：eigenvalue sharp decay 不必然成立，依赖 data distribution 的
  effective predictive rank。
- 实证部分：Fig. energy spectrum 证明多个 benchmark 上有 sharp decay。
- Limitation/Future work：当 predictive spectrum 不集中时，PDTM 的 Top-K
  efficiency advantage 会减弱，需要 adaptive K 或 online basis update。

**Manuscript change**

- Section 3.2 / Energy Concentration 段落修改措辞：
  - 将 "providing a solid theoretical justification" 改成更精确的
    "providing an objective-level justification for selecting the largest
    predictive components; the sharp decay itself is data-dependent."
- 增加 Proposition/Remark：
  - Under rank-K projection, Top-K eigenvectors maximize retained predictive
    covariance energy.
  - Retained ratio = $\sum_{i=1}^K \lambda_i / \sum_{i=1}^r \lambda_i$。
- Limitations 中补一句：若数据的 predictive spectrum flat，则需要更大的 K 或
  adaptive selection。

**Experiment need**

不需要训练，但建议补充 cumulative energy ratios across datasets/horizons 的小表。

**Response draft**

```text
We thank the reviewer for raising this important theoretical point. We agree
that eigenvalue decay should not be presented as a universal guarantee. In the
revised manuscript, we clarified the theoretical statement. What follows from
the PDTM objective is that the eigenvalues quantify the contribution of each
predictive direction to the trace maximization, and selecting the Top-K
directions is optimal for retaining the largest predictive covariance energy
under a rank-K constraint. However, a sharp decay of these eigenvalues is a
data-dependent property rather than a universal theorem. We have revised the
wording accordingly, added the retained-energy formulation, and discussed the
case of a flatter predictive spectrum as a limitation that may require a larger
or adaptively selected K.
```

**预计后续工作**

- 写入 manuscript：是。
- 写入 response letter：是。
- 写入 list of changes：是。
- 新实验：否；建议补 energy ratio table。

### Reviewer #3.6 Anonymized code repository and parameter settings

**原始意见**

> It is recommended that the authors provide a link to an anonymised code
> repository in the revised manuscript, clearly indicating the parameter
> settings, to verify the authenticity of the MACs and Memory computation
> results.

**分类与优先级**

- 类型：Major / reproducibility
- 优先级：最高
- 策略：Accept + Compliance

**Required action**

必须与当前 Phase 4 code availability route 对齐。当前 manuscript 已写入
`https://github.com/ChenhYing0v0/pdt`，但 reviewer 明确要求 anonymized code
repository。建议：

1. 创建 anonymized/private-for-review repository 或匿名压缩包链接。
2. 在 repository 中包含：
   - `baselines/PDT/` implementation；
   - `experiments/stage1/pdt/*.json` parameter manifests；
   - MACs/memory profiling script 或说明；
   - environment versions；
   - figure/table reproduction instructions。
3. 在 manuscript Code availability 和 Response #3.6 中替换为匿名 review link。
4. 如果最终选择 public GitHub，应在 response 中解释 KBS revision 已 conditionally
   accepted，是否仍需要 anonymization；但更稳妥是提供 anonymized link。

**Manuscript change**

- `Code availability` section 替换为 anonymized repository URL。
- `Reproducibility statement` 和 Word `Data_and_Code_Availability_Statement.docx`
  同步更新。
- Response letter 中说明 parameter settings are provided via manifests。

**Experiment need**

不需要新训练，但需要 repository/package verification。

**Response draft**

```text
We thank the reviewer for this suggestion. We agree that providing code and
parameter settings is important for verifying the reported MACs and memory
results. We have prepared an anonymized repository for review and added the
link to the revised manuscript. The repository contains the PDT implementation,
the manifest-based parameter configurations used for the reported experiments,
and the scripts/instructions for reproducing the MACs and memory profiling
results.
```

**预计后续工作**

- 写入 manuscript：是。
- 写入 response letter：是。
- 写入 list of changes：是。
- 新实验：否。
- 外部确认：需要用户确认 anonymized/private-for-review route。

## Cross-comment execution plan

### 必须新增或补充的 manuscript sections

1. `Conceptual comparison of predictive-domain transform`：
   - address Reviewer #1.1。
2. `Intuition of PDTM derivation`：
   - address Reviewer #1.2。
3. `Summary of symbols`：
   - address Reviewer #1.3。
4. `Hyperparameter selection and sensitivity`：
   - address Reviewer #1.4 and #3.2。
5. `Robustness to injected anomalies`：
   - address Reviewer #3.1。
6. `Analysis of learned MCD mask`：
   - address Reviewer #1.5。
7. `Additional metrics and graph baseline comparison`：
   - address Reviewer #3.3 and #3.4。
8. `Theoretical boundary of predictive energy concentration`：
   - address Reviewer #3.5。
9. `Code availability / parameter settings`：
   - address Reviewer #3.6。

### 新增实验优先级

| Priority | Task | Comments addressed | Minimum deliverable |
|---|---|---|---|
| P0 | Anomaly injection robustness | R3.1 | table with clean/corrupt/degradation ratio |
| P0 | Hyperparameter sensitivity | R1.4, R3.2 | K and threshold curves/table |
| P1 | MCD mask visualization | R1.5 | heatmap + mask density/alignment stats |
| P1 | RMSE/RSE aggregation | R3.3 | supplementary metric table |
| P1 | GNN baseline comparison | R3.4 | at least one fair graph-learning baseline |
| P2 | Energy retained-ratio table | R3.5 | cumulative predictive energy ratios |

### Response letter 写入顺序

1. Opening paragraph：感谢 reviewer，说明新增实验、理论澄清、可复现材料。
2. Reviewer #1 comments 1-5。
3. Reviewer #3 comments 1-6。
4. Summary of major changes：
   - added conceptual/theoretical clarifications；
   - added symbol table；
   - added robustness and sensitivity experiments；
   - added mask interpretability analysis；
   - added RMSE/relative metrics and graph-baseline discussion/results；
   - added anonymized code repository and parameter manifests。

### List of changes 建议条目

```text
1. Expanded the motivation and related-work discussion to distinguish PDTM from
   fixed-basis and decorrelation-oriented adaptive transforms.
2. Added derivation intuition connecting PDTM to supervised reduced-rank
   regression and clarifying its relation to CCA.
3. Added a symbol table and revised methodology notation for consistency.
4. Added hyperparameter selection guidelines and sensitivity analysis for r,
   Top-K components, and MCD threshold parameters.
5. Added quantitative anomaly-injection robustness tests.
6. Added visualization and quantitative analysis of the learned MCD channel mask.
7. Added RMSE and normalized/relative metrics to supplement MSE and MAE.
8. Added graph neural forecasting baseline discussion and comparison where the
   benchmark setting supports fair evaluation.
9. Clarified the theoretical boundary of predictive energy concentration and
   added retained predictive-energy formulation.
10. Added an anonymized code repository link and parameter manifests for
    reproducing reported experiments, MACs, and memory profiling results.
```

## 需要用户确认的事项

1. Code release route：是否创建 anonymized/private-for-review repository。
2. GNN baseline scope：是否接受 "graph-learning baselines without predefined topology"
   作为主要补充；是否额外做 traffic-only topology baseline。
3. 新实验资源：是否可在 2026-05-21 前完成 robustness、sensitivity 和 GNN baseline。
4. RMSE/relative metrics：主文增加完整三指标表，还是主文简述 + supplementary table。
5. 是否允许新增 appendix/supplementary material；如果 KBS submission 不方便新增 appendix，
   则应把主要新增结果压缩为 2 个主文表/图。

## 外部候选参考

以下仅作为 Reviewer #3.4 graph baseline discussion 的候选参考，需要在真正写入
`ref.bib` 前再次核对 BibTeX：

- MTGNN: `Connecting the Dots: Multivariate Time Series Forecasting with Graph
  Neural Networks`，arXiv:2005.11650。
- StemGNN: `Spectral Temporal Graph Neural Network for Multivariate Time-series
  Forecasting`。
- AGCRN: `Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting`。
- HR-DHAN: `Hybrid-Relation Dynamic Hypergraph Attention Network for Traffic
  Flow Prediction`，当前 `ref.bib` 已有条目。

