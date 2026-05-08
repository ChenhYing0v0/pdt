#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from protocol.runners.common import build_cli_args


METRIC_KEYS = ("mse", "rmse", "mae", "mape", "mspe")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_index_path(raw_path: str, repo_root: Path) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    marker = "artifacts/"
    normalized = raw_path.replace("\\", "/")
    if marker in normalized:
        candidate = repo_root / normalized.split(marker, 1)[1]
        candidate = repo_root / marker.rstrip("/") / normalized.split(marker, 1)[1]
        if candidate.exists():
            return candidate
    return path


def _read_index(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _discover_clean_rows(clean_runs_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run_dir in sorted(clean_runs_root.glob("r3_1_clean_*_etth1_s2023_pl*")):
        if not run_dir.is_dir():
            continue
        rows.append(
            {
                "run_id": run_dir.name,
                "checkpoint": str(run_dir / "best.ckpt"),
                "metrics": str(run_dir / "metrics.json"),
                "config": str(run_dir / "config.snapshot.json"),
                "git_meta": str(run_dir / "git_meta.json"),
                "train_log": str(run_dir / "train.log"),
            }
        )
    missing = [row["run_id"] for row in rows if not Path(row["checkpoint"]).exists() or not Path(row["config"]).exists()]
    if missing:
        raise FileNotFoundError(f"Discovered clean runs with missing checkpoint/config: {', '.join(missing)}")
    return rows


def _load_clean_rows(index_path: Path, clean_runs_root: Path) -> list[dict[str, str]]:
    if index_path.exists():
        return _read_index(index_path)
    rows = _discover_clean_rows(clean_runs_root)
    if not rows:
        raise FileNotFoundError(
            f"Clean index not found at {index_path}, and no clean runs were discovered under {clean_runs_root}."
        )
    return rows


def _model_order(model: str) -> int:
    return {"PDT": 0, "iTransformer": 1, "DLinear": 2}.get(model, 99)


def _run_sort_key(row: dict[str, str]) -> tuple[int, int]:
    snapshot = _read_json(_resolve_index_path(row["config"], REPO_ROOT))
    args = snapshot["manifest"]["args"]
    return _model_order(str(args["model"])), int(args["pred_len"])


def _resolve_root_path(args: dict[str, Any], data_root: str | None) -> None:
    if not data_root:
        return
    if args.get("data") == "ETTh1":
        args["root_path"] = str(Path(data_root).expanduser() / "ETT-small")


def _eval_run_id(clean_run_id: str, corruption_type: str) -> str:
    return f"r3_1_{corruption_type}_{clean_run_id.removeprefix('r3_1_clean_')}"


def _build_eval_command(
    repo_root: Path,
    row: dict[str, str],
    output_root: Path,
    corruption_type: str,
    corruption_rate: float,
    corruption_amp: float,
    corruption_seed: int,
    segment_len: int,
    data_root: str | None,
) -> tuple[str, Path, list[str]]:
    clean_run_id = row["run_id"]
    snapshot = _read_json(_resolve_index_path(row["config"], repo_root))
    manifest = snapshot["manifest"]
    args = dict(manifest["args"])
    _resolve_root_path(args, data_root)

    eval_run_id = _eval_run_id(clean_run_id, corruption_type)
    output_dir = output_root / "eval_runs" / corruption_type / eval_run_id
    args.update(
        {
            "is_training": 0,
            "checkpoint_path": str(_resolve_index_path(row["checkpoint"], repo_root)),
            "output_dir": str(output_dir),
            "run_id": eval_run_id,
            "test_corruption_type": corruption_type,
            "test_corruption_rate": corruption_rate,
            "test_corruption_amp": corruption_amp,
            "test_corruption_seed": corruption_seed,
            "test_corruption_segment_len": segment_len,
            "skip_predictions": True,
        }
    )

    command = [sys.executable, "-u", str(repo_root / "baselines/PDT/run.py"), *build_cli_args(args)]
    return eval_run_id, output_dir, command


def _run_command(repo_root: Path, output_dir: Path, command: list[str], gpu: str | None, dry_run: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(" ".join(command))
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}".rstrip(":")
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = gpu

    with (output_dir / "eval.log").open("w", encoding="utf-8") as log_handle:
        subprocess.run(command, cwd=repo_root, env=env, stdout=log_handle, stderr=subprocess.STDOUT, check=True)


def _collect_results(index_rows: list[dict[str, str]], output_root: Path, corruption_types: list[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in sorted(index_rows, key=_run_sort_key):
        clean_run_id = row["run_id"]
        snapshot = _read_json(_resolve_index_path(row["config"], REPO_ROOT))
        args = snapshot["manifest"]["args"]
        model = str(args["model"])
        pred_len = int(args["pred_len"])
        clean_metrics = _read_json(_resolve_index_path(row["metrics"], REPO_ROOT))

        for corruption_type in corruption_types:
            eval_run_id = _eval_run_id(clean_run_id, corruption_type)
            corrupt_path = output_root / "eval_runs" / corruption_type / eval_run_id / "metrics.json"
            if not corrupt_path.exists():
                continue
            corrupt_metrics = _read_json(corrupt_path)
            result: dict[str, Any] = {
                "dataset": str(args["data"]),
                "model": model,
                "pred_len": pred_len,
                "clean_run_id": clean_run_id,
                "eval_run_id": eval_run_id,
                "corruption_type": corruption_type,
            }
            for key in METRIC_KEYS:
                clean_value = float(clean_metrics[key])
                corrupt_value = float(corrupt_metrics[key])
                result[f"clean_{key}"] = clean_value
                result[f"corrupt_{key}"] = corrupt_value
                result[f"delta_{key}_pct"] = ((corrupt_value - clean_value) / clean_value * 100.0) if clean_value else 0.0
            results.append(result)
    return results


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["model"]), str(row["corruption_type"]))
        grouped.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for (dataset, model, corruption_type), group in sorted(grouped.items(), key=lambda item: (item[0][0], _model_order(item[0][1]), item[0][2])):
        out: dict[str, Any] = {
            "dataset": dataset,
            "model": model,
            "corruption_type": corruption_type,
            "num_horizons": len(group),
        }
        for key in ("mse", "rmse", "mae"):
            out[f"avg_clean_{key}"] = statistics.fmean(float(row[f"clean_{key}"]) for row in group)
            out[f"avg_corrupt_{key}"] = statistics.fmean(float(row[f"corrupt_{key}"]) for row in group)
            out[f"avg_delta_{key}_pct"] = statistics.fmean(float(row[f"delta_{key}_pct"]) for row in group)
        summary.append(out)
    return summary


def _write_latex_table(path: Path, summary: list[dict[str, Any]]) -> None:
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for row in summary:
        by_model.setdefault(str(row["model"]), {})[str(row["corruption_type"])] = row

    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Model & Clean MSE & Spike MSE & Spike $\\Delta$MSE & Segment MSE & Segment $\\Delta$MSE \\\\",
        "\\midrule",
    ]
    for model in sorted(by_model, key=_model_order):
        spike = by_model[model].get("spike", {})
        segment = by_model[model].get("segment", {})
        clean_mse = spike.get("avg_clean_mse", segment.get("avg_clean_mse", float("nan")))
        lines.append(
            f"{model} & {clean_mse:.4f} & "
            f"{spike.get('avg_corrupt_mse', float('nan')):.4f} & "
            f"{spike.get('avg_delta_mse_pct', float('nan')):.2f}\\% & "
            f"{segment.get('avg_corrupt_mse', float('nan')):.4f} & "
            f"{segment.get('avg_delta_mse_pct', float('nan')):.2f}\\% \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run R3.1 test-time corruption evaluation.")
    parser.add_argument("--clean-index", default="artifacts/revision/r3_1_noise_robustness/clean_checkpoints/index.tsv")
    parser.add_argument("--clean-runs-root", default=os.environ.get("CLEAN_RUNS_ROOT") or os.environ.get("OUTPUT_ROOT") or "artifacts/runs")
    parser.add_argument("--output-root", default="artifacts/revision/r3_1_noise_robustness")
    parser.add_argument("--corruption-types", default="spike,segment")
    parser.add_argument("--corruption-rate", type=float, default=0.05)
    parser.add_argument("--corruption-amp", type=float, default=3.0)
    parser.add_argument("--corruption-seed", type=int, default=2023)
    parser.add_argument("--segment-len", type=int, default=4)
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT"))
    parser.add_argument("--gpu", default=os.environ.get("GPU"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-missing", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()

    repo_root = REPO_ROOT
    index_rows = _load_clean_rows((repo_root / args.clean_index).resolve(), (repo_root / args.clean_runs_root).resolve())
    output_root = (repo_root / args.output_root).resolve()
    corruption_types = [item.strip() for item in args.corruption_types.split(",") if item.strip()]

    if not args.collect_only:
        for row in sorted(index_rows, key=_run_sort_key):
            for corruption_type in corruption_types:
                eval_run_id, output_dir, command = _build_eval_command(
                    repo_root=repo_root,
                    row=row,
                    output_root=output_root,
                    corruption_type=corruption_type,
                    corruption_rate=args.corruption_rate,
                    corruption_amp=args.corruption_amp,
                    corruption_seed=args.corruption_seed,
                    segment_len=args.segment_len,
                    data_root=args.data_root,
                )
                if args.only_missing and (output_dir / "metrics.json").exists():
                    print(f"skip existing {eval_run_id}")
                    continue
                print(f"== Evaluating {eval_run_id} ==")
                _run_command(repo_root, output_dir, command, args.gpu, args.dry_run)

    if not args.dry_run:
        rows = _collect_results(index_rows, output_root, corruption_types)
        summary = _summarize(rows)
        _write_csv(output_root / "robustness_runs.csv", rows)
        _write_csv(output_root / "robustness_summary.csv", summary)
        _write_latex_table(output_root / "robustness_table.tex", summary)
        print(f"Wrote {len(rows)} per-run rows to {output_root / 'robustness_runs.csv'}")
        print(f"Wrote {len(summary)} summary rows to {output_root / 'robustness_summary.csv'}")


if __name__ == "__main__":
    main()
