# KBS 返修剩余任务清单

更新日期：2026-05-07

## 已完成

- Phase 0：revision workspace 与 package manifest。
- Phase 1：Word-format submission scaffolds。
- Phase 2：LaTeX revised manuscript source package。
- Phase 3：PNG figures、DPI metadata、LaTeX figure reference 校准。
- Phase 4：data/code availability 与 reproducibility statement 初稿。
- 当前 `List_of_Changes.docx` 已补齐 Phase 0-4 所有已完成变更；reviewer-specific changes 将在逐条处理 reviewer comments 后追加。

## 仍需完成但需要外部确认

### 1. Code release route

当前状态：

- Manuscript 中已写入 `https://github.com/ChenhYing0v0/pdt` 作为待确认 public release URL。
- Reviewer #3 提到 anonymized code repository。

需要确认：

- 返修前直接公开 `https://github.com/ChenhYing0v0/pdt`？
- 或创建 anonymized/private-for-review link？
- 或仅写 post-acceptance release？

影响：

- `source_latex/elsarticle-template-num.tex` 中 `Code availability` 的最终措辞。
- `Response_to_Reviewers.docx` 中 Reviewer #3 comment 6 的回复。

### 2. Declaration of interest

当前状态：

- Manuscript 内声明：no known competing financial interests or personal relationships。
- 新建 `Declaration_of_Interest.docx` 也按 no competing interests 起草。
- 原始 `manuscript/PDT_ver01_mst/declarationStatement.docx` 的勾选状态与上述文字可能不一致。

需要确认：

- 最终是否确认为 no competing interests？
- 是否需要重新生成/替换 journal declaration form？

### 3. Language editing

当前状态：

- `Revised_Cover_Letter.docx` 已预留 language editing statement。

需要确认：

- 使用 Elsevier Author Services？
- 使用其他 professional editing？
- 或由作者内部完成 English polishing？

### 4. Editorial Manager item types

当前状态：

- 用户已确认 revised manuscript 最终上传 LaTeX。
- Package 当前保留 `source_latex/` 与 `figures_png/` 的相对路径关系。

需要确认：

- Editorial Manager 是否保留目录结构？
- 是否要求把 `.tex`、`.bib`、`.bst` 和 `.png` 全部放在同一层？
- 是否要求 tracked-changes manuscript copy？

## 仍需完成且属于 reviewer-comment 工作

以下任务需要进入 reviewer comments 逐条讨论后再做：

- Phase 5 逐条 action matrix 已形成独立文档：
  `manuscript/revision/PDT_revision_v01/checks/phase5_reviewer_comment_action_matrix.md`。
  该文档已根据 2026-05-08 用户审阅反馈更新为 user-reviewed execution plan，
  并拆分为“需要重跑实验或等待实验结果的任务”和“仅需文字修改、证明、合规准备的任务”。
  后续实际正文/Word 文件修改应以该文档为入口。
- Reviewer #1.1：PDTM motivation 与 FreDF/TransDF 概念对比。
- Reviewer #1.2：PDTM derivation intuition，CCA / low-rank regression 解释。
- Reviewer #1.3：notation consistency 与 symbol table。
- Reviewer #1.4：rank、Top-K、MCD mask hyperparameters。
- Reviewer #1.5：MCD learned mask interpretability。
- Reviewer #3.1：noise/anomaly injection robustness tests。实验意图与最小设计已落地到
  `manuscript/revision/PDT_revision_v01/checks/r3_1_noise_anomaly_robustness_experiment_design.md`；
  当前执行范围已收敛为 ETTh1 全 horizon，ECL 因训练耗时和单 `pred_len` 展示问题暂缓。
  clean checkpoint manifests、DLinear protocol runner、远程训练脚本和回传归档脚本已落地。
  后续仍需在远程训练机运行 clean checkpoint 训练，再按该设计补 test-time corruption
  评估代码、运行实验并回填结果。
- Reviewer #3.2：key hyperparameter sensitivity。
- Reviewer #3.3：补充 RMSE 或相对误差指标。
- Reviewer #3.4：graph neural network baselines。
- Reviewer #3.5：Top-K predictive information concentration 的理论边界。
- Reviewer #3.6：anonymized code repository 与参数设置说明。

## 建议下一步

1. 先确认 code release route 和 declaration of interest。
2. 按 Phase 5 action matrix 并行推进：实验侧优先 robustness、Top-K/mask threshold
   sensitivity、GNN baselines 和 MCD mask interpretability；文字侧优先 FreDF/TransDF
   深读、Appendix proof、symbol table、Top-K theory boundary 和 anonymized code
   repository 方案。
3. 最后填充 `Response_to_Reviewers.docx`，并把 reviewer-specific changes 追加到 `List_of_Changes.docx`，同步修改 LaTeX manuscript。
