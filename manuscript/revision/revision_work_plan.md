# KBS 返修工作计划

更新日期：2026-05-07

## 范围

本文档用于规划以下论文第一次返修所需的提交合规、文件准备和最终打包工作：

- Manuscript ID：`KNOSYS-D-26-03771`
- 论文题目：`PDT: Predictive Domain Transform for Multivariate Time Series Forecasting`
- 期刊：`Knowledge-Based Systems`
- 当前状态：conditionally accepted pending revision
- 返修截止日期：2026-05-26

本文档暂不逐条分析 reviewer comments。审稿意见的逐条处理、实验取舍和 rebuttal 策略将在后续单独讨论。

## 当前执行状态

- Phase 0 已完成：已创建 `manuscript/revision/PDT_revision_v01/`，并建立
  `source_latex/`、`word_files/`、`figures_original_pdf/` 和 `checks/`。
- Phase 1 已完成 scaffold：已生成 7 个 Word-format submission files。
- Phase 2 已完成初始 package 准备：用户已确认 revised manuscript 最终上传 LaTeX，
  已建立 `source_latex/` 主路径。
- Phase 3 已完成图片插入校准：已使用 `figures_png/` 中的 PNG figures，统一写入
  约 330 dpi metadata，并将 LaTeX active figure references 从 PDF 改为 PNG。
- Phase 4 已完成初稿：已更新 data availability，新增 code availability，并生成
  Word-format availability / reproducibility statements。
- 详细完成记录见：
  `manuscript/revision/PDT_revision_v01/checks/phase0_1_2_completion_log.md`。
- Phase 3 详细记录见：
  `manuscript/revision/PDT_revision_v01/checks/phase3_figure_completion_log.md`。
- Phase 4 详细记录见：
  `manuscript/revision/PDT_revision_v01/checks/phase4_availability_completion_log.md`。
- 当前剩余任务清单见：
  `manuscript/revision/PDT_revision_v01/checks/remaining_revision_tasks.md`。
- File-by-file package manifest 见：
  `manuscript/revision/PDT_revision_manifest.md`。

## 已确认要求

### 来自 editor letter 的硬性要求

- 需要针对每条审稿意见提供 point-by-point response 或 list of changes。
- 返修前需要进行 language editing；editor letter 明确建议使用 Elsevier Author Services 的 English Language Editing service。
- Revised cover letter、response to reviewers、highlights、revised manuscript、CRediT author statement、author agreement 和 declaration of interest 均应准备为 Word format。
- 只提交一个最终版本的 manuscript。
- 如果使用 LaTeX，需要同时提交 source file 和 supporting `.bib` files。
- Revised manuscript 中的 tables、figures 和 equations 应保持 editable format。
- Figures 需要达到 300 dpi 的 good quality；editor letter 明确要求 figures 不应使用 PDF format。
- 可选：如需 interactive figures，可将 MATLAB `.fig` 文件作为 supplementary material 上传。

### 来自当前 KBS / Elsevier guidance 的要求

- KBS 要求整套 submission 提供 editable source files；PDF 不能作为 source file。
- LaTeX 返修提交需要包含相关 editable source files。
- Highlights 应作为独立 editable file 提交，文件名中应包含 `highlights`；KBS guidance 要求 3 到 5 条 bullet，每条不超过 85 characters。
- Equations 和 tables 必须是 editable text，不能是图片。
- Figures 应作为单独文件提交，命名清晰，并按 manuscript 中的引用顺序组织。
- KBS 当前 artwork guidance 一般允许 vector drawings 使用 EPS/PDF，但本次 editor letter 对 figures 的要求更严格，因此本计划优先遵守 editor letter：返修 package 中不使用 PDF figure files 作为最终图文件。
- KBS 要求提供 research data availability statement。当前 data policy 路径为 Option C：存储并引用/链接 research data，或说明为何无法共享。
- 需要提供 CRediT contribution roles。
- Editorial Manager 的 revision 流程可能要求单独上传 response document，也可能根据系统配置要求 tracked-changes manuscript copy。正式提交前必须检查并批准系统生成的最终 PDF。

## 本地 manuscript 状态

`manuscript/` 下已确认的文件：

- 主 LaTeX source：
  `manuscript/PDT_ver01_mst/elsarticle-template-num.tex`
- Bibliography：
  `manuscript/PDT_ver01_mst/ref.bib`
- 当前编译 manuscript PDF：
  `manuscript/PDT_ver01_mst/build/elsarticle-template-num.pdf`
- 当前 cover letter：
  `manuscript/PDT_ver01_mst/CoverLetter.pdf`
- 当前 author agreement：
  `manuscript/PDT_ver01_mst/author_agreement.pdf`
- 当前 declaration statement：
  `manuscript/PDT_ver01_mst/declarationStatement.docx`
- 当前 figure files，均为 PDF format：
  `energy.pdf`、`overview.pdf`、`embedding.pdf`、`visual.pdf`、
  `efficiency.pdf`、`prefer.pdf`、`lookback.pdf`、`representation.pdf`
- Editor letter：
  `manuscript/revision/editor_letter.md`

当前直接缺口：

- 尚无 response-to-reviewers document。
- 尚无 list-of-changes document。
- Highlights 目前嵌在 LaTeX 中，尚无单独 Word highlights file。
- CRediT 和 declaration text 在 manuscript 内已有内容，但尚无单独 Word CRediT statement。
- Cover letter 和 author agreement 当前为 PDF，而 editor letter 要求 Word format。
- 当前 manuscript folder 内所有 figure assets 均为 PDF，而 editor letter 要求 non-PDF、300 dpi figures。
- 尚无专门的 revision package manifest。
- 尚未确认 Editorial Manager 本轮是否要求 tracked-changes manuscript copy。

## 工作计划

### Phase 0：冻结输入并创建 revision workspace

目标日期：2026-05-07

- 保留原始投稿 manuscript folder 作为 evidence baseline，不直接覆盖。
- 创建专门的 revision workspace，例如：
  `manuscript/revision/PDT_revision_v01/`。
- 将原始 LaTeX source、bibliography 和 figure assets 复制到或引用到 revision workspace。
- 新增 revision package manifest，逐项列出待上传文件、file type、source path 和当前状态。
- 在 manifest 顶部记录 editor letter 的硬性要求，作为最终 checklist 的依据。

交付物：

- `manuscript/revision/PDT_revision_v01/`
- `manuscript/revision/PDT_revision_manifest.md`

### Phase 1：准备 submission-compliance 文件

目标日期：2026-05-07 至 2026-05-09

- 起草 Word-format revised cover letter。
- 起草 Word-format response-to-reviewers scaffold。本阶段只搭建结构，不在逐条讨论 reviewer comments 前写详细回复。
- 起草 Word-format list-of-changes scaffold。
- 将 highlights 导出为单独 Word file，并按 3 到 5 条 bullet、每条不超过 85 characters 的要求压缩。
- 创建单独 Word CRediT author statement。
- 更新或重建 Word declaration of interest。
- 如 submission system 不接受当前 PDF，则重建 Word author agreement。

交付物：

- `Revised_Cover_Letter.docx`
- `Response_to_Reviewers.docx`
- `List_of_Changes.docx`
- `Highlights.docx`
- `CRediT_Author_Statement.docx`
- `Declaration_of_Interest.docx`
- `Author_Agreement.docx`

### Phase 2：准备 manuscript revision package

目标日期：2026-05-09 至 2026-05-13

- 用户已确认 revised manuscript 最终上传 LaTeX；`source_latex/` 是本轮 revised manuscript 的主路径。
- 保留 LaTeX source package：
  `elsarticle-template-num.tex`、`math_utils.tex`、`ref.bib` 和
  `elsarticle-num.bst`。
- 确保 equations 为 editable LaTeX，而不是图片。
- 确保 tables 仍为 editable LaTeX/Word tables，而不是截图。
- 添加或更新必要声明：
  data availability、code availability、declaration of competing interest、CRediT、acknowledgements，以及如适用的 generative-AI disclosure。
- 在后续完成 reviewer-response planning 后，在 revision workspace 中应用 manuscript edits，并编译 clean review PDF 供内部检查。

交付物：

- Revised manuscript source package。
- 内部检查用 compiled PDF。
- 如 Editorial Manager 要求，则准备 tracked-changes copy。

### Phase 3：处理 figure 与 artwork compliance

目标日期：2026-05-10 至 2026-05-15

- 查找或重生成当前八个 PDF figures 的原始 source。
- 按 editor letter 要求，将每个 final figure 导出为 non-PDF artwork，优先使用 300 dpi 或更高分辨率的 PNG/TIFF。
- 对 line drawings 或 line/halftone 混合图，在可行时使用 500 dpi 或更高分辨率，因为 KBS artwork guidance 对 line art 和 combination artwork 有更高分辨率建议。
- 使用清晰命名，例如 `Figure_1_energy.png`、`Figure_2_overview.png`。
- 确认 figure order 与 manuscript 中的引用顺序一致。
- 保持 figure captions 为 manuscript 中的 editable text，不只嵌入图片内部。
- 如果存在 MATLAB `.fig` source，且作者希望提供 interactive figures，则准备可选 supplementary `.fig` uploads；否则在 manifest 中记录为 not used。

交付物：

- Non-PDF final figures。
- Figure source archive 或 regeneration script notes。
- Figure order 与 DPI checklist。

### Phase 4：准备 data、code 与 reproducibility statements

目标日期：2026-05-12 至 2026-05-16

- 确认所有 benchmark datasets 是否公开，以及论文是否能引用其 official repositories。
- 添加满足 KBS Option C 的 data availability statement：提供 repository links，或明确说明无法共享的原因。
- 决定 code release 路径：
  anonymized repository、private-for-review link，或 post-acceptance release。
- 如果 manuscript 中放置 code link，需要确保 parameter settings、MACs、memory profiling scripts 和 environment notes 可被审稿方找到。
- 如果代码以可引用形式公开，添加 software citation 或 repository reference。

交付物：

- Data availability statement。
- Code availability statement。
- 可选 anonymized code repository link。

### Phase 5：reviewer comments 处理块

目标日期：2026-05-15 至 2026-05-21

本阶段暂缓，后续单独讨论。

预计后续工作：

- 将每条 reviewer comment 拆分为 required action、manuscript change、experiment need 和 response strategy。
- 判断截止日前哪些新增实验可行、哪些只能通过分析或文字澄清处理。
- 按 point-by-point 结构更新 response letter 和 list of changes。
- 应用 manuscript edits，并补充 reviewer-response work 产生的新 figures/tables。
- 对新增实验或新增 reported metrics 运行最小必要验证。

### Phase 6：最终打包与提交

目标日期：2026-05-22 至 2026-05-24

- 按 editor letter 和 KBS guide 运行最终 file checklist。
- 编译 LaTeX manuscript 并检查 PDF。
- 完成 language editing 后再次检查 spelling/grammar。
- 检查 text 中引用的 references 是否均出现在 `ref.bib`，以及 `ref.bib` 中条目是否被正确引用。
- 检查每个 table、figure 和 equation 是否按顺序引用和编号。
- 在 Editorial Manager 中上传所有文件。
- 生成系统 PDF。
- 仔细检查系统生成的 revision PDF。
- 如生成 PDF 出现 file order、figure、font、equation 或 metadata 问题，在批准前回到 revision package 修正。
- 只有在系统生成 PDF 正确后才批准并正式提交 revision。

建议内部截止日期：2026-05-24，即比 2026-05-26 期刊截止日提前两天。

## 立即行动项

1. 创建 `PDT_revision_v01/` 和 package manifest。
2. 将 editor letter 的硬性要求转换为 file-by-file checklist。
3. 准备 Word-format scaffolds：cover letter、response letter、list of changes、highlights、CRediT statement、declaration of interest 和 author agreement。
4. 查找当前八个 PDF figures 的 source files 或 regeneration scripts。
5. 在最终打包前确认 Editorial Manager 本轮 revision 的实际 item types。

## 已查询来源

- KBS Guide for Authors：
  https://www.sciencedirect.com/journal/knowledge-based-systems/publish/guide-for-authors
- Elsevier Editorial Manager revision instructions：
  https://www.elsevier.support/publishing/answer/how-do-i-revise-my-submission-in-editorial-manager
- Elsevier highlights guidance：
  https://www.elsevier.support/publishing/answer/how-do-i-include-highlights-with-my-manuscript
