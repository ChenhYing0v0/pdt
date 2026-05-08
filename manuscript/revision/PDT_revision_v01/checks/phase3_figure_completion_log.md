# Phase 3 图片与 artwork compliance 完成记录

更新日期：2026-05-07

## 本轮目标

按 Phase 3 处理 `Knowledge-Based Systems` 返修中的 figure compliance：

- 使用已生成的 PNG figures。
- 不使用 PDF figures 作为最终 figure upload。
- 不存在 MATLAB `.fig` source，因此不准备 interactive figure supplementary files。
- 对 manuscript 只进行图片插入校准，不进行任何正文、表格、公式或 reviewer-response 改动。

## 已完成内容

### 1. PNG figures 检查与 DPI metadata 校准

已确认 `manuscript/revision/PDT_revision_v01/figures_png/` 下存在 8 个 PNG：

| Figure | PNG file | Pixel size | DPI metadata | 状态 |
|---|---|---:|---:|---|
| Energy spectrum | `energy.png` | 4534 x 1234 | 330 x 330 | 已完成 |
| Architecture overview | `overview.png` | 4945 x 2722 | 330 x 330 | 已完成 |
| Embedding comparison | `embedding.png` | 1659 x 838 | 330 x 330 | 已完成 |
| Visual analysis | `visual.png` | 5215 x 2598 | 330 x 330 | 已完成 |
| Efficiency trade-off | `efficiency.png` | 1226 x 766 | 330 x 330 | 已完成 |
| Architecture preference | `prefer.png` | 4262 x 1485 | 330 x 330 | 已完成 |
| Look-back sensitivity | `lookback.png` | 4381 x 1439 | 330 x 330 | 已完成 |
| Representation visualization | `representation.png` | 5545 x 1727 | 330 x 330 | 已完成 |

说明：

- `efficiency.png` 原始 DPI metadata 约为 144。
- `energy.png` 原始文件未写入 DPI metadata。
- 本轮只补齐/统一 PNG metadata 到 330 dpi，不改变图像像素尺寸和图像内容。

### 2. Manuscript 图片插入校准

已修改：

- `manuscript/revision/PDT_revision_v01/source_latex/elsarticle-template-num.tex`

具体改动：

- 新增 `\graphicspath{{../figures_png/}}`，使 revision LaTeX source 直接引用 `figures_png/`。
- 将 8 个 `\includegraphics` 从 PDF 文件改为 PNG 文件：
  - `energy.pdf` -> `energy.png`
  - `overview.pdf` -> `overview.png`
  - `embedding.pdf` -> `embedding.png`
  - `visual.pdf` -> `visual.png`
  - `efficiency.pdf` -> `efficiency.png`
  - `prefer.pdf` -> `prefer.png`
  - `lookback.pdf` -> `lookback.png`
  - `representation.pdf` -> `representation.png`

未改动：

- 未改正文表述。
- 未改 figure caption。
- 未改 table、equation、reference、data availability、code availability。
- 未处理 reviewer comments。

### 3. MATLAB `.fig` source

用户确认不存在 MATLAB `.fig` source。

处理结果：

- 不准备 `.fig` supplementary files。
- Manifest 中将 optional interactive figure 标记为 not used。

## 验证记录

### PNG metadata 验证

使用 bundled Python + PIL 检查 `figures_png/`，8 个 PNG 均显示约 `329.9968 x 329.9968` DPI metadata。

### LaTeX 引用验证

使用 `rg` 检查 `source_latex/elsarticle-template-num.tex`：

- 已存在 `\graphicspath{{../figures_png/}}`。
- 8 个 active `\includegraphics` 均引用 `.png`。
- 除被注释的 graphical abstract 示例外，active manuscript figure references 不再引用 `.pdf`。

### 编译验证

在 `manuscript/revision/PDT_revision_v01/source_latex/` 下运行：

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build elsarticle-template-num.tex
```

结果：

- 编译成功。
- 生成内部检查 PDF：
  `manuscript/revision/PDT_revision_v01/source_latex/build/elsarticle-template-num.pdf`
- `.fls` 记录显示 8 个 PNG 均被 LaTeX 读取：
  `energy.png`、`overview.png`、`embedding.png`、`visual.png`、
  `efficiency.png`、`prefer.png`、`lookback.png`、`representation.png`。

保留 warning：

- 原稿中已有的 overfull hbox、BibTeX warning 和 missing character warning 仍存在。
- 本轮按用户要求只做图片插入校准，未处理这些正文/排版 warning。

## 后续注意

- 最终上传时需确保 `source_latex/` 与 `figures_png/` 的相对路径关系被保留，或在正式打包前按 Editorial Manager 要求调整为同目录/压缩包路径。
- 如果 Editorial Manager 不保留目录结构，需在提交前把 PNG 文件复制到 LaTeX source 同级目录，并同步更新 `\graphicspath` 或 `\includegraphics` 路径。
