# KBS 返修 Package Manifest

更新日期：2026-05-07

## 基本信息

- Manuscript ID：`KNOSYS-D-26-03771`
- 论文题目：`PDT: Predictive Domain Transform for Multivariate Time Series Forecasting`
- 期刊：`Knowledge-Based Systems`
- 返修截止日期：2026-05-26
- Revision workspace：`manuscript/revision/PDT_revision_v01/`
- Revised manuscript 最终上传路径：LaTeX source package

## Editor Letter 硬性要求对照

| 要求 | 当前处理 | 状态 |
|---|---|---|
| Point-by-point response 或 list of changes | 已创建 `Response_to_Reviewers.docx`；`List_of_Changes.docx` 已补齐当前 Phase 0-4 已完成变更 | Response 待逐条讨论；list of changes 当前阶段已补齐 |
| Language editing | 已在 cover letter scaffold 中预留 language editing statement | 待完成 |
| Revised cover letter 为 Word format | 已创建 `Revised_Cover_Letter.docx` | 已完成 scaffold |
| Response to reviewers 为 Word format | 已创建 `Response_to_Reviewers.docx` | 已完成 scaffold |
| Highlights 为 Word format | 已创建 `Highlights.docx`，4 条 bullet 均小于 85 characters | 已完成 scaffold |
| Revised manuscript 为 Word format / source file 可 Word 或 LaTeX | 用户已确认最终上传 LaTeX；已创建 `source_latex/` | Phase 2 已落实 LaTeX 路径 |
| CRediT author statement 为 Word format | 已创建 `CRediT_Author_Statement.docx` | 已完成 scaffold |
| Author agreement 为 Word format | 已创建 `Author_Agreement.docx` | 已完成 scaffold，需作者签署/确认 |
| Declaration of interest 为 Word format | 已创建 `Declaration_of_Interest.docx` | 已完成 scaffold，需作者确认冲突声明 |
| LaTeX 需包含 source 和 `.bib` | 已复制 `.tex`、`math_utils.tex`、`.bib`、`.bst` 到 `source_latex/` | 已完成 |
| Tables、figures、equations editable | LaTeX equations/tables 保持 editable；figures 已切换为 PNG artwork | 已完成当前阶段 |
| Figures 300 dpi 且不为 PDF | 已使用 `figures_png/` 中 8 个 PNG，并统一写入约 330 dpi metadata | 已完成 |
| 只提交一个最终 manuscript version | Manifest 仅保留 LaTeX revised manuscript 路径 | 已按计划约束 |

## File-by-file Package 清单

### Revised Manuscript Source Package

| 文件 | 路径 | 用途 | 状态 |
|---|---|---|---|
| Main LaTeX source | `manuscript/revision/PDT_revision_v01/source_latex/elsarticle-template-num.tex` | Revised manuscript source | 已准备，后续承载 reviewer-driven edits |
| Math macros | `manuscript/revision/PDT_revision_v01/source_latex/math_utils.tex` | LaTeX macro dependency | 已准备 |
| Bibliography | `manuscript/revision/PDT_revision_v01/source_latex/ref.bib` | Supporting `.bib` file | 已准备 |
| Elsevier bst | `manuscript/revision/PDT_revision_v01/source_latex/elsarticle-num.bst` | Bibliography style dependency | 已准备 |

### Word-format Submission Files

| 文件 | 路径 | 用途 | 状态 |
|---|---|---|---|
| Revised cover letter | `manuscript/revision/PDT_revision_v01/word_files/Revised_Cover_Letter.docx` | Editor-facing cover letter | 已完成 scaffold |
| Response to reviewers | `manuscript/revision/PDT_revision_v01/word_files/Response_to_Reviewers.docx` | Point-by-point response | 已完成 scaffold，详细回复待后续 |
| List of changes | `manuscript/revision/PDT_revision_v01/word_files/List_of_Changes.docx` | Summary of changes | 已补齐当前 Phase 0-4 已完成变更；reviewer-specific changes 待后续追加 |
| Highlights | `manuscript/revision/PDT_revision_v01/word_files/Highlights.docx` | Separate highlights file | 已完成，4 条 bullet 均小于 85 characters |
| CRediT author statement | `manuscript/revision/PDT_revision_v01/word_files/CRediT_Author_Statement.docx` | Author contributions | 已完成 scaffold |
| Declaration of interest | `manuscript/revision/PDT_revision_v01/word_files/Declaration_of_Interest.docx` | Competing interest declaration | 已完成 scaffold，需作者最终确认 |
| Author agreement | `manuscript/revision/PDT_revision_v01/word_files/Author_Agreement.docx` | Author agreement | 已完成 scaffold，需作者签署/确认 |
| Data and code availability statement | `manuscript/revision/PDT_revision_v01/word_files/Data_and_Code_Availability_Statement.docx` | Editable availability statement | 已完成初稿，release route 待确认 |
| Reproducibility statement | `manuscript/revision/PDT_revision_v01/word_files/Reproducibility_Statement.docx` | Environment/config/artifact说明 | 已完成初稿 |

### Figure Files

| Figure | Final PNG path | Pixel size | DPI metadata | 状态 |
|---|---|---|---|
| Energy spectrum | `manuscript/revision/PDT_revision_v01/figures_png/energy.png` | 4534 x 1234 | 330 x 330 | 已完成 |
| Architecture overview | `manuscript/revision/PDT_revision_v01/figures_png/overview.png` | 4945 x 2722 | 330 x 330 | 已完成 |
| Embedding comparison | `manuscript/revision/PDT_revision_v01/figures_png/embedding.png` | 1659 x 838 | 330 x 330 | 已完成 |
| Visual analysis | `manuscript/revision/PDT_revision_v01/figures_png/visual.png` | 5215 x 2598 | 330 x 330 | 已完成 |
| Efficiency trade-off | `manuscript/revision/PDT_revision_v01/figures_png/efficiency.png` | 1226 x 766 | 330 x 330 | 已完成 |
| Architecture preference | `manuscript/revision/PDT_revision_v01/figures_png/prefer.png` | 4262 x 1485 | 330 x 330 | 已完成 |
| Look-back sensitivity | `manuscript/revision/PDT_revision_v01/figures_png/lookback.png` | 4381 x 1439 | 330 x 330 | 已完成 |
| Representation visualization | `manuscript/revision/PDT_revision_v01/figures_png/representation.png` | 5545 x 1727 | 330 x 330 | 已完成 |

### Optional Interactive Figures

- MATLAB `.fig` source：不存在。
- 处理状态：不准备 `.fig` supplementary files。

## 已确认决策

- Revised manuscript 最终按 LaTeX 上传，不再将 Word manuscript 作为主路径。
- `manuscript/PDT_ver01_mst/` 作为原始投稿 baseline 保留，不直接覆盖。
- Phase 1 的 response 和 list of changes 只建立结构，不提前替用户决定 reviewer-response 策略。
- Phase 3 对 manuscript 仅进行图片插入校准，不做正文、caption、table、equation 或 reviewer-response 改动。

## 待确认事项

- Editorial Manager 是否仍要求 tracked-changes manuscript copy。
- Language editing 的实际完成方式和 cover letter 中的表述。
- Declaration of interest 的最终勾选项：当前 manuscript 声明无 competing interests，但原始 `declarationStatement.docx` 的勾选内容需要作者复核。
- Figure source 是否存在可直接重生成 PNG/TIFF 的脚本或原始工程文件。
- Code availability statement 是否采用 anonymized repository、private-for-review link，还是 post-acceptance release。
- 当前 manuscript 已写入 `https://github.com/ChenhYing0v0/pdt` 作为待确认 public release URL；最终提交前需确认仓库是否公开或是否替换为 anonymized review link。
- Editorial Manager 是否保留 LaTeX source 与 `figures_png/` 的目录结构；若不保留，需在最终打包前调整图像路径。
