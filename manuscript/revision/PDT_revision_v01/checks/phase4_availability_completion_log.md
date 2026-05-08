# Phase 4 Data / Code / Reproducibility 完成记录

更新日期：2026-05-07

## 本轮目标

完成 Phase 4 中可以在本地直接推进的事项：

- 明确 manuscript 中的 data availability statement。
- 新增 code availability statement。
- 生成 Word-format availability / reproducibility statements。
- 记录 reviewer-facing code release 和 final package 的待确认项。

本轮仍不逐条处理 reviewer comments。

## 已完成内容

### 1. Manuscript statements

已修改：

- `manuscript/revision/PDT_revision_v01/source_latex/elsarticle-template-num.tex`

具体改动：

- 将原 `Data will be made available on request.` 改为更具体的 public benchmark dataset availability 表述。
- 新增 `\section*{Code availability}`。
- Code availability 中暂使用当前仓库 remote 对应的 public URL 形式：
  `https://github.com/ChenhYing0v0/pdt`。

当前 manuscript statement：

```tex
\section*{Data availability}
The benchmark datasets used in this study are publicly available from their original sources, as described in Section 4.1. The experimental scripts expect the standard dataset folders, including ETT-small, weather, electricity, traffic, and exchange-rate data.

\section*{Code availability}
The source code and experiment configuration files are being prepared for public release at \url{https://github.com/ChenhYing0v0/pdt}. The repository contains the PDT implementation, manifest-based experiment configuration files, and scripts for launching and validating the reported experiments.
```

### 2. Word-format statements

已新增：

- `manuscript/revision/PDT_revision_v01/word_files/Data_and_Code_Availability_Statement.docx`
- `manuscript/revision/PDT_revision_v01/word_files/Reproducibility_Statement.docx`

用途：

- `Data_and_Code_Availability_Statement.docx`：作为 submission package 中可复用的 editable availability statement。
- `Reproducibility_Statement.docx`：记录 implementation environment、manifest configuration 和 artifact verification 说明，便于后续补入 response 或 supplementary material。

### 3. 本地依据

已检查：

- Git remote：
  `git@github.com:ChenhYing0v0/pdt.git`
- README 中已有 dry-run 与 remote execution 说明。
- Repository 中存在 `experiments/stage1/pdt/` manifest files。
- Manuscript 当前实验设置已写明 `PyTorch 2.5.1` 和 `NVIDIA RTX 3090 GPU`。

## 待确认事项

- `https://github.com/ChenhYing0v0/pdt` 是否将在返修前公开。
- Reviewer #3 提到 anonymized code repository；如果本轮返修需要匿名链接，应在逐条 rebuttal 前确认 anonymized/private-for-review route。
- 当前 statement 说 `being prepared for public release`，最终提交前需要根据真实 release 状态改为 `available at` 或 `will be made available`。
- Dataset 是否只引用 original sources，不随代码包重新分发。

## 验证记录

### LaTeX 编译

已在 `manuscript/revision/PDT_revision_v01/source_latex/` 下运行：

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build elsarticle-template-num.tex
```

结果：

- 编译成功。
- 内部检查 PDF 已更新：
  `manuscript/revision/PDT_revision_v01/source_latex/build/elsarticle-template-num.pdf`
- 检查 log 中未发现 `LaTeX Error`、`Fatal error`、缺失 PNG 或图片读取失败。

保留 warning：

- 原稿已有的 overfull hbox、BibTeX warning、missing character warning 仍存在。
- 本轮不处理正文/排版 warning。

### Word 文件结构检查

已用 `python-docx` 打开两个新增文件：

- `Data_and_Code_Availability_Statement.docx`：10 个有效 paragraph，1 个 metadata table。
- `Reproducibility_Statement.docx`：13 个有效 paragraph，1 个 metadata table。
