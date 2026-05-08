# Phase 0/1/2 完成记录

更新日期：2026-05-07

## 本轮目标

按 `manuscript/revision/revision_work_plan.md` 完成 Phase 0、Phase 1 和 Phase 2：

- Phase 0：冻结输入并创建 revision workspace。
- Phase 1：准备 submission-compliance 文件。
- Phase 2：准备 manuscript revision package，且 revised manuscript 最终上传 LaTeX。

本轮不逐条处理 reviewer comments。

## Phase 0 完成内容

已创建 revision workspace：

- `manuscript/revision/PDT_revision_v01/`

已建立子目录：

- `source_latex/`：存放最终上传的 LaTeX revised manuscript source package。
- `word_files/`：存放 Word-format submission scaffolds。
- `figures_original_pdf/`：存放当前 PDF figure inputs，供 Phase 3 重生成非 PDF figures。
- `checks/`：存放检查记录、完成记录和待确认项。

已复制 LaTeX source package 初始文件：

- `source_latex/elsarticle-template-num.tex`
- `source_latex/math_utils.tex`
- `source_latex/ref.bib`
- `source_latex/elsarticle-num.bst`

已复制当前 PDF figure inputs：

- `figures_original_pdf/energy.pdf`
- `figures_original_pdf/overview.pdf`
- `figures_original_pdf/embedding.pdf`
- `figures_original_pdf/visual.pdf`
- `figures_original_pdf/efficiency.pdf`
- `figures_original_pdf/prefer.pdf`
- `figures_original_pdf/lookback.pdf`
- `figures_original_pdf/representation.pdf`

已创建 package manifest：

- `manuscript/revision/PDT_revision_manifest.md`

## Phase 1 完成内容

已创建 7 个 Word-format scaffolds：

- `word_files/Revised_Cover_Letter.docx`
- `word_files/Response_to_Reviewers.docx`
- `word_files/List_of_Changes.docx`
- `word_files/Highlights.docx`
- `word_files/CRediT_Author_Statement.docx`
- `word_files/Declaration_of_Interest.docx`
- `word_files/Author_Agreement.docx`

Word scaffolds 的内容来源：

- `Revised_Cover_Letter.docx`：参考原始 `CoverLetter.pdf` 的提交语境，改为 revised submission scaffold，并预留 language editing statement。
- `Response_to_Reviewers.docx`：按 editor comment、Reviewer #1、Reviewer #3 建立 point-by-point response 框架；每条只放 reviewer comment 摘要和待填写 response/manuscript change，不做策略判断。
- `List_of_Changes.docx`：记录本轮 package-level changes，并预留后续 manuscript changes 表格。
- `Highlights.docx`：从 LaTeX highlights 压缩为 4 条独立 bullet，均小于 85 characters。
- `CRediT_Author_Statement.docx`：根据 manuscript 内 CRediT section 创建。
- `Declaration_of_Interest.docx`：根据 manuscript 内 `Declaration of Competing Interest` 创建。
- `Author_Agreement.docx`：根据原始 `author_agreement.pdf` 创建 editable scaffold，并预留 signature/date 栏。

已完成的结构性检查：

- 7 个 `.docx` 均可被 `python-docx` 打开。
- `Highlights.docx` 中 4 条 bullet 的字符数分别为 63、71、70、63，满足 KBS 每条不超过 85 characters 的要求。

未完成的 visual QA：

- 已尝试使用 Documents skill 的 `render_docx.py` 渲染 `Revised_Cover_Letter.docx`。
- 当前环境缺少 `soffice`，因此无法完成 DOCX-to-PNG visual render QA。
- 本轮只完成结构性检查；最终提交前仍建议在具备 Word/LibreOffice 的环境中打开并人工检查版式。

## Phase 2 完成内容

已落实 revised manuscript 的最终上传决策：

- 用户确认 revised manuscript 最终上传 LaTeX。
- 因此本轮建立 `source_latex/` 作为唯一 revised manuscript 主路径。

已准备 LaTeX source package 初始版本：

- `source_latex/elsarticle-template-num.tex`
- `source_latex/math_utils.tex`
- `source_latex/ref.bib`
- `source_latex/elsarticle-num.bst`

已确认当前 manuscript 中的 editable 状态：

- Equations：当前为 LaTeX source，保持 editable。
- Tables：当前为 LaTeX tables，保持 editable。
- Figures：当前 source 中仍引用 PDF figures，最终不满足 editor letter；此项转入 Phase 3 处理。

已记录需要后续 manuscript edits 的位置：

- Reviewer-driven scientific revisions：待后续逐条讨论 reviewer comments。
- Data availability：当前 manuscript 为 `Data will be made available on request.`，需确认是否改为 repository link、on request，或其他 KBS Option C 表述。
- Code availability：当前 manuscript 未单独提供 code availability statement；需结合 anonymized code repository 决策后补充。
- Declaration of competing interest：manuscript 表述为无 competing interests；原始 Word declaration 的勾选状态需作者确认。
- Tracked-changes manuscript：是否需要取决于 Editorial Manager 本轮 item types，尚未确认。

## 将进行的更改

下一阶段开始前，建议优先推进以下更改：

1. 在 `source_latex/elsarticle-template-num.tex` 中按后续 reviewer-response 策略修改 manuscript。
2. 根据最终 data/code release 决策，更新 data availability 和 code availability statements。
3. 将所有 figure references 从当前 PDF 输入迁移到 Phase 3 生成的 non-PDF final figures。
4. 在 `Response_to_Reviewers.docx` 和 `List_of_Changes.docx` 中逐条填入 response、manuscript location 和 evidence。
5. 根据 language editing 完成情况更新 `Revised_Cover_Letter.docx`。

## 风险与待确认

- `declarationStatement.docx` 的勾选状态与 manuscript 内 declaration 文字可能不一致，需要作者复核。
- 当前环境无法完成 DOCX visual render QA，需要后续用 Word/LibreOffice 打开检查。
- Figure source 尚未定位，Phase 3 可能需要从实验脚本或绘图脚本重新导出 PNG/TIFF。
- 如果 Editorial Manager 强制要求 Word revised manuscript，需要另行从 LaTeX 转 Word；当前用户确认的主路径是 LaTeX。
