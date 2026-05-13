#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_json(path: Path) -> dict[str, Any]:
    return _load_json(path) if path.exists() else {}


def _resolve_path(raw_path: str) -> Path:
    return Path(os.path.expandvars(raw_path)).expanduser()


def _manifest_from_run(run_dir: Path) -> dict[str, Any]:
    snapshot = _load_json(run_dir / "config.snapshot.json")
    manifest = snapshot.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError(f"Missing manifest in {run_dir / 'config.snapshot.json'}")
    return manifest


def _mask_2d(mask_path: Path) -> np.ndarray:
    mask = np.load(mask_path).astype(np.float64)
    while mask.ndim > 2:
        mask = mask.mean(axis=0)
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError(f"Expected square channel mask, got shape {mask.shape} from {mask_path}")
    return mask


def _resolve_dataset_csv(args: dict[str, Any], data_root: str | None) -> Path:
    raw_root = _resolve_path(str(args["root_path"]))
    data_path = str(args["data_path"])
    if data_root:
        local_root = _resolve_path(data_root)
        candidates = [
            local_root / raw_root.name / data_path,
            local_root / data_path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return raw_root / data_path


def _feature_frame(manifest: dict[str, Any], data_root: str | None) -> pd.DataFrame:
    args = manifest["args"]
    csv_path = _resolve_dataset_csv(args, data_root)
    raw = pd.read_csv(csv_path)
    features = str(args.get("features", "M"))
    target = str(args.get("target", "OT"))
    if features in {"M", "MS"}:
        cols = [col for col in raw.columns if col != "date"]
        if target in cols:
            cols = [col for col in cols if col != target] + [target]
        return raw[cols]
    return raw[[target]]


def _test_slice(values: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    args = manifest["args"]
    data_name = str(args.get("data", "")).lower()
    seq_len = int(args["seq_len"])
    n = len(values)
    if data_name in {"etth1", "etth2"}:
        start = 12 * 30 * 24 + 4 * 30 * 24 - seq_len
        end = 12 * 30 * 24 + 8 * 30 * 24
    elif data_name in {"ettm1", "ettm2"}:
        start = 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len
        end = 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4
    else:
        num_train = int(n * 0.7)
        num_test = int(n * 0.2)
        num_vali = n - num_train - num_test
        start = n - num_test - seq_len
        end = n
        _ = num_vali
    return values.iloc[max(start, 0):min(end, n)]


def _abs_corr(values: pd.DataFrame) -> np.ndarray:
    corr = values.corr(method="pearson").abs().fillna(0.0).to_numpy(dtype=np.float64)
    np.fill_diagonal(corr, 0.0)
    return corr


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    if len(a) < 3:
        return float("nan")
    return _pearson(_rankdata(a), _rankdata(b))


def _offdiag_values(matrix: np.ndarray) -> np.ndarray:
    keep = ~np.eye(matrix.shape[0], dtype=bool)
    return matrix[keep]


def _plot_heatmaps(mask: np.ndarray, corr: np.ndarray | None, output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    panels = 2 if corr is not None and corr.shape == mask.shape else 1
    width = 7.2 if panels == 2 else 3.6
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })
    fig, axes = plt.subplots(1, panels, figsize=(width, 3.2), constrained_layout=True)
    if panels == 1:
        axes = [axes]
    cmap = "viridis"
    image = axes[0].imshow(mask, cmap=cmap, vmin=0, vmax=1, aspect="equal", interpolation="nearest")
    axes[0].set_title("(a) MCD mask", pad=4)
    axes[0].set_xlabel("Channel")
    axes[0].set_ylabel("Channel")
    cbar = fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.03)
    cbar.set_ticks([0.0, 0.5, 1.0])
    if panels == 2 and corr is not None:
        corr_image = axes[1].imshow(corr, cmap=cmap, vmin=0, vmax=1, aspect="equal", interpolation="nearest")
        axes[1].set_title("(b) Future correlation", pad=4)
        axes[1].set_xlabel("Channel")
        axes[1].set_ylabel("Channel")
        corr_cbar = fig.colorbar(corr_image, ax=axes[1], fraction=0.046, pad=0.03)
        corr_cbar.set_ticks([0.0, 0.5, 1.0])
    for ax in axes:
        ax.tick_params(length=2.5, width=0.6)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
        if mask.shape[0] > 15:
            ticks = np.arange(0, mask.shape[0], 4)
            if ticks[-1] != mask.shape[0] - 1:
                ticks = np.append(ticks, mask.shape[0] - 1)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
    _ = title
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _read_run(run_dir: Path, output_dir: Path, data_root: str | None) -> dict[str, Any]:
    manifest = _manifest_from_run(run_dir)
    if manifest.get("stage") != "revision_mcd_mask_interpretability":
        raise ValueError(f"Not an MCD mask interpretability run: {run_dir}")
    if data_root:
        os.environ["DATA_ROOT"] = data_root

    mask = _mask_2d(run_dir / "channel_mask.npy")
    mask_stats = _maybe_json(run_dir / "mask_stats.json")
    metrics = _maybe_json(run_dir / "metrics.json")

    corr = None
    pearson = float("nan")
    spearman = float("nan")
    try:
        features = _feature_frame(manifest, data_root)
        test_values = _test_slice(features, manifest)
        corr = _abs_corr(test_values)
        if corr.shape == mask.shape:
            pearson = _pearson(_offdiag_values(mask), _offdiag_values(corr))
            spearman = _spearman(_offdiag_values(mask), _offdiag_values(corr))
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"warning: cannot compute data correlation for {run_dir.name}: {exc}")

    dataset = str(manifest["dataset"])
    pred_len = int(manifest["pred_len"])
    figure_path = output_dir / "figures" / f"{run_dir.name}_mask.png"
    _plot_heatmaps(mask, corr, figure_path, f"{dataset}, pred_len={pred_len}")

    density = float(mask_stats.get("density_gt_0_5", (mask > 0.5).mean()))
    row = {
        "run_id": run_dir.name,
        "dataset": dataset,
        "pred_len": pred_len,
        "channels": mask.shape[0],
        "k_top": manifest["args"].get("k_top"),
        "mask_threshold": manifest["args"].get("mask_threshold"),
        "mse": metrics.get("mse"),
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
        "mask_mean": float(mask.mean()),
        "mask_std": float(mask.std()),
        "density_gt_0_5": density,
        "alignment_pearson": pearson,
        "alignment_spearman": spearman,
        "figure": str(figure_path),
    }
    return row


def _score(row: dict[str, Any]) -> tuple[float, float, float]:
    channels = float(row["channels"])
    density = float(row["density_gt_0_5"])
    align = row["alignment_spearman"]
    align_value = float(align) if align == align else -1.0
    readability = -abs(channels - 21.0)
    sparsity_balance = -abs(density - 0.35)
    return (readability, align_value, sparsity_balance)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize and visualize MCD mask interpretability candidates.")
    parser.add_argument("--runs-root", default="artifacts/runs", help="Root containing protocol run directories.")
    parser.add_argument("--output-dir", default="artifacts/analysis/mcd_mask_interpretability")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT"), help="Dataset root for correlation reference.")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    run_dirs = sorted(
        path for path in runs_root.glob("mcd_mask_*")
        if path.is_dir() and (path / "config.snapshot.json").exists() and (path / "channel_mask.npy").exists()
    )
    if not run_dirs:
        raise FileNotFoundError(f"No completed mcd_mask_* runs with channel_mask.npy under {runs_root}")

    rows = [_read_run(run_dir, output_dir, args.data_root) for run_dir in run_dirs]
    rows.sort(key=_score, reverse=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    recommendation_path = output_dir / "recommendation.md"
    best = rows[0]
    recommendation_path.write_text(
        "\n".join(
            [
                "# MCD Mask Interpretability Candidate Ranking",
                "",
                f"Best current candidate: `{best['run_id']}`",
                "",
                "Ranking favors readable channel count, stronger mask/correlation alignment, and non-trivial sparsity.",
                "",
                f"- summary: `{summary_path}`",
                f"- best figure: `{best['figure']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")
    print(f"Wrote {recommendation_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
