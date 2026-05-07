# R_2026_PDT AGENTS

This file contains only the rules that are actually effective for the current
repo.

Cross-project defaults remain in `~/.codex/AGENTS.md`.
The preserved heavier Scholar reference lives at
`docs/AGENTS.scholar-reference.md`.

## Session Start

At the first substantive turn in this repo:

1. Run:
   `python3 ~/.codex/scripts/codex_hook_emulation.py session-start --cwd "$PWD"`
2. Check `git status` and note branch plus local changes.
3. Identify the minimal relevant skills for the task.
4. Summarize the current repo state and Obsidian binding before deeper edits.

This repo does not rely on a local `scripts/codex_hook_emulation.py`.
Always use the global helper under `~/.codex/scripts/`.

## Project Facts

- Repo root:
  `/Users/river/PaperResearch/Project/R_2026_PDT`
- Bound vault:
  `/Users/river/Obsidian/ResearchVault/Research/r-2026-pdt`
- Repo-local project memory:
  `.codex/project-memory/r-2026-pdt.md`
- Binding registry:
  `.codex/project-memory/registry.yaml`
- Note language: `zh-CN`
- Current maturity: bootstrap stage; research question, datasets, and first
  experiment line are still to be defined.

## Communication And Decision Style

- Use Chinese in responses and keep technical terms in English.
- Unless explicitly requested otherwise, write analysis reports in Chinese.
- Keep tone direct, concrete, and audit-oriented.
- Separate confirmed facts from inference explicitly.
- For code-grounded model explanations, describe tensor transformations through
  tensor names, shapes, and operations before giving high-level interpretation.
- For complex work, prefer plan-first when a simpler route may exist.
- When intent is unclear, ask first; before important operations, confirm first.
- Clean up temporary files and temporary artifacts after the task when
  practical.
- When a rule from the preserved Scholar reference conflicts with the current
  repo workflow, follow this file.

## Project Bootstrap And Knowledge Base

- If `.codex/project-memory/registry.yaml` exists, default to
  `obsidian-project-memory` and treat Obsidian as the durable knowledge sink for
  this project.
- If the repo is still unbound but clearly represents this research project,
  use `obsidian-project-bootstrap` first.
- For substantive research turns, update today's `Daily/` and repo-local
  project memory; update `00-Hub.md` only when top-level project status
  changes.
- Keep the vault small and canonical:
  `00-Hub.md`, `01-Plan.md`, `Knowledge/`, `Papers/`, `Experiments/`,
  `Results/Reports/`, `Writing/`, `Daily/`, `Archive/`.
- Do not mirror the whole repository into the vault.
- Obsidian workflow does not require MCP or extra API keys.

## Literature And Note Workflow

- Treat Zotero as the source of truth for paper discovery and metadata.
- Write or update canonical project paper notes under the bound project's
  `Papers/`.
- Default note language is Chinese unless the user explicitly changes it.
- Keep the paper-note section contract from `0. 翻译摘要原文` through `6. 总结`.
- Render formulas with `$...$` or `$$...$$`, never backticks.
- Before making strong summary claims, verify whether the PDF or full text is
  complete; if not, state the missing scope and lower confidence.
- For research-judgment tasks, analyze first and wait for confirmation before
  writing `Knowledge/` unless the user explicitly asks for direct writeback.
- The source of truth for `paper-summarize` is the global skill under
  `~/.codex/skills/paper-summarize/`; do not create a drifting repo-local copy
  unless the user explicitly asks for one.

## Repo Evolution Style

- This repo is new. Prefer minimal, reversible structure changes over premature
  framework scaffolding.
- Prefer shell-first, repo-first changes before introducing custom tooling.
- Avoid assuming a fixed model stack, trainer framework, dataset layout, or
  remote runbook before the project defines them explicitly.
- Only add code skeletons, configs, or experiment wrappers when they map to a
  concrete research need.

## Model And Analysis Documentation

- Every model-code version update must synchronously create or update a
  code-facing explanation document under `docs/`.
- Explanatory project documents should be written in Chinese by default, while
  keeping code identifiers, tensor names, module names, metrics, and established
  technical terms in English.
- For non-model code updates, organize explanations by functional module, such
  as training, data loading, metrics, runner, diagnostics, remote scripts, or
  analysis. Do not force a line-by-line walkthrough when module-level structure
  is clearer.
- For model-structure updates, organize the explanation by the actual forward
  computation flow. Describe tensor names, shapes, operations, and where each
  changed tensor enters downstream modules before giving high-level
  interpretation.
- For non-trivial or high-risk local logic inside a functional module, include a
  tight line-range walkthrough with the relevant shape or artifact effects. Use
  line ranges as evidence, not as the primary document structure.
- After each model implementation, add a code-theory consistency evaluation:
  state the intended theory, how the code realizes it, what remains only a
  proxy, and what evidence would falsify the design.
- Analysis scripts and stats appendices must define every new statistic, CSV
  column, and figure quantity by source tensor/file, computation, and meaning.
- New concepts, abbreviations, metrics, and claims must be defined before they
  are used as evidence.

## Verification Boundary

When files change, prefer the smallest honest verification chain that matches
the current repo state:

1. File-existence or format checks for newly created project structure.
2. `python -m py_compile` on touched Python files.
3. JSON or YAML parse checks for touched config files.
4. Targeted dry-runs or tests only after the repo defines them.

Do not claim end-to-end experiment success unless training or evaluation
actually ran.

## Experiment Reproducibility

- When experiments begin, set seeds for `random`, `numpy`, `torch`,
  `torch.cuda`, and `PYTHONHASHSEED` when reproducibility matters.
- Record the effective config at run start.
- Record Python version, torch version, CUDA version, GPU model, and dataset
  identity when experimental conclusions depend on them.
- Do not call a result reproducible until the artifacts needed to rerun it
  actually exist.

## Git Preference

- When a commit is requested, use Conventional Commits.
- Do not use destructive commands such as `git push --force`,
  `git reset --hard`, or deleting tracked history unless the user explicitly
  asks.

## Hook And Risk Control

Before risky or irreversible actions, run:

`python3 ~/.codex/scripts/codex_hook_emulation.py preflight "<command>"`

After meaningful edits, run:

`python3 ~/.codex/scripts/codex_hook_emulation.py post-edit --cwd "$PWD"`

If the helper requests caution, follow it.

## Session Wrap-Up Protocol

When the user says `wrap up`, `总结`, `session end`, or similar:

1. Run:
   `python3 ~/.codex/scripts/codex_hook_emulation.py session-end --cwd "$PWD"`
2. Generate a work log summarizing what was accomplished.
3. Check whether `AGENTS.md` needs updates based on changes made.
4. Remind about any temporary files that should be cleaned up.
5. Show `git status` for uncommitted changes.

## 任务完成总结

每次任务完成时，主动提供简要总结：

```text
📋 本次操作回顾
1. [主要操作]
2. [修改的文件]

📊 当前状态
• [Git/文件系统/运行状态]

💡 下一步建议
1. [针对性建议]
```

## Skill Routing

Prefer the smallest matching set:

- `obsidian-project-memory` for durable project-state updates
- `obsidian-project-bootstrap` for initial binding or rebuild
- `obsidian-literature-workflow` for paper-note and synthesis work
- `paper-summarize` for structured paper reading notes
- `results-analysis` and `results-report` for completed experiment bundles
- `doc-coauthoring` for substantial documentation rewrites
- `bug-detective` for bug reports
