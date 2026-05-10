#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "experiments/revision/r3_2_hyperparam_sensitivity"
PRED_LEN = 336
TOPK_GRID = [8, 16, 32, 64, 96]
THRESHOLD_GRID = [0.05, 0.10, 0.15, 0.25]

DATASETS = [
    {
        "slug": "etth1",
        "base_manifest": "experiments/stage1/pdt/etth1_multi_pred_len.json",
        "rrr_dir": "ETTh1",
        "rrr_prefix": "ETTh1",
    },
    {
        "slug": "weather",
        "base_manifest": "experiments/stage1/pdt/weather_multi_pred_len.json",
        "rrr_dir": "weather",
        "rrr_prefix": "weather",
    },
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _select_index(values: list[int], pred_len: int) -> int:
    try:
        return values.index(pred_len)
    except ValueError as exc:
        raise ValueError(f"pred_len={pred_len} is not present in {values}") from exc


def _scalarize_args(args: dict[str, Any], index: int) -> dict[str, Any]:
    scalar_args: dict[str, Any] = {}
    for key, value in args.items():
        if isinstance(value, list):
            scalar_args[key] = value[index]
        else:
            scalar_args[key] = value
    return scalar_args


def _rk_mat_file(rrr_dir: str, rrr_prefix: str, k_top: int, pred_len: int) -> str:
    return f"dataset/RRR_mats/{rrr_dir}/{rrr_prefix}_RRR_L96_R{k_top}_H{pred_len}_R.npy"


def _variant_manifest(
    base: dict[str, Any],
    args: dict[str, Any],
    slug: str,
    axis: str,
    value: int | float,
    default_k: int,
    default_threshold: float,
    rrr_dir: str,
    rrr_prefix: str,
) -> tuple[str, dict[str, Any]]:
    manifest = copy.deepcopy(base)
    manifest["stage"] = "revision_r3_2_hyperparam_sensitivity"
    manifest["pred_len"] = PRED_LEN
    manifest["search_budget"] = {
        "trials": 1,
        "policy": "fixed_reviewer_sensitivity",
    }

    variant_args = copy.deepcopy(args)
    if axis == "topk":
        k_top = int(value)
        threshold = default_threshold
        suffix = f"topk_k{k_top:03d}"
        variant_args["k_top"] = k_top
        variant_args["rk_mat_file"] = _rk_mat_file(rrr_dir, rrr_prefix, k_top, PRED_LEN)
        variant_args["mask_threshold"] = threshold
        variant_args["des"] = f"R32TopK{k_top:03d}"
    elif axis == "mask_threshold":
        k_top = default_k
        threshold = float(value)
        suffix = f"mask_tau{int(round(threshold * 100)):03d}"
        variant_args["k_top"] = k_top
        variant_args["rk_mat_file"] = _rk_mat_file(rrr_dir, rrr_prefix, k_top, PRED_LEN)
        variant_args["mask_threshold"] = threshold
        variant_args["des"] = f"R32Tau{int(round(threshold * 100)):03d}"
    else:
        raise ValueError(f"Unknown sensitivity axis: {axis}")

    variant_args["pred_len"] = PRED_LEN
    manifest["args"] = variant_args
    manifest["sensitivity"] = {
        "axis": axis,
        "value": value,
        "default_k_top": default_k,
        "default_mask_threshold": default_threshold,
        "pred_len": PRED_LEN,
    }

    filename = f"{slug}_pl{PRED_LEN}_{suffix}.json"
    return filename, manifest


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index_rows = ["manifest\tdataset\tpred_len\taxis\tvalue\tdefault_k_top\tdefault_mask_threshold"]
    generated: list[Path] = []

    for spec in DATASETS:
        base_path = REPO_ROOT / spec["base_manifest"]
        base = _read_json(base_path)
        pred_lens = base["pred_len"]
        if not isinstance(pred_lens, list):
            raise TypeError(f"{base_path} must be a multi-pred_len manifest")
        pred_index = _select_index(pred_lens, PRED_LEN)
        args = _scalarize_args(base["args"], pred_index)
        default_k = int(args["k_top"])
        default_threshold = float(args["mask_threshold"])

        variants: list[tuple[str, int | float]] = [("topk", k_top) for k_top in TOPK_GRID]
        variants.extend(("mask_threshold", threshold) for threshold in THRESHOLD_GRID if threshold != default_threshold)

        for axis, value in variants:
            filename, manifest = _variant_manifest(
                base=base,
                args=args,
                slug=str(spec["slug"]),
                axis=axis,
                value=value,
                default_k=default_k,
                default_threshold=default_threshold,
                rrr_dir=str(spec["rrr_dir"]),
                rrr_prefix=str(spec["rrr_prefix"]),
            )
            output_path = OUTPUT_DIR / filename
            _write_json(output_path, manifest)
            generated.append(output_path)
            index_rows.append(
                "\t".join(
                    [
                        filename,
                        str(base["dataset"]),
                        str(PRED_LEN),
                        axis,
                        str(value),
                        str(default_k),
                        str(default_threshold),
                    ]
                )
            )

    (OUTPUT_DIR / "manifest_index.tsv").write_text("\n".join(index_rows) + "\n", encoding="utf-8")
    print(f"Generated {len(generated)} manifests under {OUTPUT_DIR.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
