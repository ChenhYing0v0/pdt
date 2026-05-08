from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDNAMES = [
    "run_id",
    "baseline",
    "stage",
    "dataset",
    "seq_len",
    "pred_len",
    "seed",
    "selection_policy",
    "metric_policy",
    "mse",
    "mae",
    "rmse",
    "mape",
    "mspe",
    "rse",
    "corr",
]


def iter_run_rows(runs_root: Path):
    # Recursively scan for metrics files so nested phase folders are supported.
    for metrics_path in sorted(runs_root.rglob("metrics.json"), key=lambda p: p.as_posix()):
        run_dir = metrics_path.parent
        config_path = run_dir / "config.snapshot.json"
        if not config_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        manifest = config["manifest"]
        yield {
            "run_id": run_dir.name,
            "baseline": manifest["baseline"],
            "stage": manifest["stage"],
            "dataset": manifest["dataset"],
            "seq_len": manifest["seq_len"],
            "pred_len": manifest["pred_len"],
            "seed": manifest["seed"],
            "selection_policy": manifest["selection_policy"],
            "metric_policy": manifest["metric_policy"],
            "mse": metrics.get("mse"),
            "mae": metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "mape": metrics.get("mape"),
            "mspe": metrics.get("mspe"),
            "rse": metrics.get("rse"),
            "corr": metrics.get("corr"),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect protocol run metrics into a CSV file.")
    parser.add_argument("--runs-root", required=True, help="Directory containing protocol run folders.")
    parser.add_argument("--output", required=True, help="CSV output path.")
    args = parser.parse_args()

    runs_root = Path(args.runs_root).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in iter_run_rows(runs_root):
            writer.writerow(row)

    print(f"Wrote summary CSV: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
