#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "experiments/revision/mcd_mask_interpretability"

DATASETS = [
    {
        "slug": "etth1",
        "base_manifest": "experiments/stage1/pdt/etth1_multi_pred_len.json",
        "pred_lens": [96, 192, 336, 720],
    },
    {
        "slug": "weather",
        "base_manifest": "experiments/stage1/pdt/weather_multi_pred_len.json",
        "pred_lens": [96, 192, 336, 720],
    },
    {
        "slug": "ecl",
        "base_manifest": "experiments/stage1/pdt/ecl_multi_pred_len.json",
        "pred_lens": [96, 192, 336, 720],
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


def _build_manifest(base: dict[str, Any], args: dict[str, Any], pred_len: int) -> dict[str, Any]:
    manifest = copy.deepcopy(base)
    manifest["stage"] = "revision_mcd_mask_interpretability"
    manifest["pred_len"] = pred_len
    manifest["search_budget"] = {
        "trials": 1,
        "policy": "fixed_mask_interpretability_candidates",
    }

    run_args = copy.deepcopy(args)
    run_args["pred_len"] = pred_len
    run_args["des"] = f"MCDMaskPL{pred_len}"
    manifest["args"] = run_args
    manifest["mask_interpretability"] = {
        "purpose": "candidate_for_main_text_series_representation_analysis",
        "pred_len": pred_len,
        "expected_outputs": [
            "metrics.json",
            "channel_mask.npy",
            "mask_stats.json",
        ],
    }
    return manifest


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index_rows = ["manifest\tdataset\tpred_len\tk_top\tmask_threshold\tenc_in"]
    generated: list[Path] = []

    for spec in DATASETS:
        base_path = REPO_ROOT / spec["base_manifest"]
        base = _read_json(base_path)
        pred_lens = base["pred_len"]
        if not isinstance(pred_lens, list):
            raise TypeError(f"{base_path} must be a multi-pred_len manifest")

        for pred_len in spec["pred_lens"]:
            pred_index = _select_index(pred_lens, pred_len)
            args = _scalarize_args(base["args"], pred_index)
            manifest = _build_manifest(base, args, pred_len)
            filename = f"{spec['slug']}_pl{pred_len}.json"
            output_path = OUTPUT_DIR / filename
            _write_json(output_path, manifest)
            generated.append(output_path)
            index_rows.append(
                "\t".join(
                    [
                        filename,
                        str(base["dataset"]),
                        str(pred_len),
                        str(args["k_top"]),
                        str(args["mask_threshold"]),
                        str(args["enc_in"]),
                    ]
                )
            )

    (OUTPUT_DIR / "manifest_index.tsv").write_text("\n".join(index_rows) + "\n", encoding="utf-8")
    print(f"Generated {len(generated)} manifests under {OUTPUT_DIR.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
