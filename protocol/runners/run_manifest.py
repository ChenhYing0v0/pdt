from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from protocol.io import build_git_meta, ensure_dir, generate_run_id, write_json, write_run_snapshot
from protocol.manifests import load_manifest, manifest_from_data
from protocol.runners.dlinear import build_command as build_dlinear_command
from protocol.runners.itransformer import build_command as build_itransformer_command
from protocol.runners.pdt import build_command as build_pdt_command


BUILDERS = {
    "dlinear": build_dlinear_command,
    "itransformer": build_itransformer_command,
    "pdt": build_pdt_command,
}


def _parse_override_value(raw_value: str):
    lowered = raw_value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered == "null":
        return None
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


def _apply_override(data: dict, dotted_key: str, raw_value: str) -> None:
    parts = dotted_key.split(".")
    current = data
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = _parse_override_value(raw_value)


def _apply_overrides(manifest, overrides: list[str]):
    if not overrides:
        return manifest
    updated = copy.deepcopy(manifest.raw)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected KEY=VALUE.")
        key, value = item.split("=", 1)
        _apply_override(updated, key, value)
    return manifest_from_data(updated, manifest.path)


def _expand_batch_manifests(manifest) -> list[tuple[Any, dict[str, Any] | None]]:
    pred_lens = manifest.pred_lens
    if isinstance(manifest.pred_len, int):
        return [(manifest, None)]

    original_manifest = copy.deepcopy(manifest.raw)
    args = original_manifest.get("args", {})
    if not isinstance(args, dict):
        raise TypeError("Manifest field 'args' must be an object.")

    aligned_args: dict[str, list[Any]] = {}
    broadcast_args: dict[str, Any] = {}
    for key, value in args.items():
        if key == "pred_len":
            continue
        if isinstance(value, list):
            if any(isinstance(item, (list, dict)) for item in value):
                raise ValueError(f"Manifest args.{key} cannot be a nested list/dict for pred_len alignment.")
            if len(value) != len(pred_lens):
                raise ValueError(
                    f"Manifest args.{key} length {len(value)} does not match pred_len length {len(pred_lens)}."
                )
            aligned_args[key] = value
        else:
            broadcast_args[key] = value

    expanded: list[tuple[Any, dict[str, Any] | None]] = []
    total = len(pred_lens)
    for index, active_pred_len in enumerate(pred_lens):
        child_raw = copy.deepcopy(original_manifest)
        child_raw["pred_len"] = active_pred_len
        child_args = dict(broadcast_args)
        for key, values in aligned_args.items():
            child_args[key] = values[index]
        child_args["pred_len"] = active_pred_len
        child_raw["args"] = child_args
        child_manifest = manifest_from_data(child_raw, manifest.path)
        batch_context = {
            "index": index,
            "size": total,
            "active_pred_len": active_pred_len,
            "batch_pred_lens": pred_lens,
        }
        expanded.append((child_manifest, batch_context))
    return expanded


def _resolve_run_id(manifest, cli_run_id: str | None, batch_size: int, active_pred_len: int) -> str:
    if cli_run_id is None:
        return generate_run_id(manifest)
    if batch_size == 1:
        return cli_run_id
    return f"{cli_run_id}_pl{active_pred_len}"


def _stream_process(command: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> int:
    merged_env = os.environ.copy()
    merged_env.update(env)
    merged_env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
        return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a baseline experiment from a protocol manifest.")
    parser.add_argument("--manifest", required=True, help="Path to the JSON manifest.")
    parser.add_argument("--run-id", help="Override the generated run id.")
    parser.add_argument("--output-root", help="Root directory for run artifacts.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override manifest values, e.g. --set pred_len=192 --set args.pred_len=192",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without executing.")
    cli_args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    manifest = load_manifest(cli_args.manifest)
    manifest = _apply_overrides(manifest, cli_args.overrides)
    builder = BUILDERS.get(manifest.baseline.lower())
    if builder is None:
        raise ValueError(f"Unsupported baseline: {manifest.baseline}")

    output_root = Path(cli_args.output_root).resolve() if cli_args.output_root else repo_root / "artifacts" / "runs"
    original_manifest = copy.deepcopy(manifest.raw)
    expanded_manifests = _expand_batch_manifests(manifest)

    for child_manifest, batch_context in expanded_manifests:
        if not isinstance(child_manifest.pred_len, int):
            raise TypeError("Expanded child manifest must have a single pred_len.")
        run_id = _resolve_run_id(
            child_manifest,
            cli_args.run_id,
            len(expanded_manifests),
            child_manifest.pred_len,
        )
        run_dir = ensure_dir(output_root / run_id)
        command, runner_env = builder(repo_root, child_manifest, run_dir)
        write_run_snapshot(
            run_dir,
            child_manifest,
            command,
            runner_env,
            original_manifest=original_manifest,
            batch_context=batch_context,
        )
        write_json(run_dir / "git_meta.json", build_git_meta(repo_root, child_manifest.baseline.lower()))

        if cli_args.dry_run:
            print("Resolved run directory:")
            print(run_dir)
            print("Resolved command:")
            print(" ".join(command))
            continue

        return_code = _stream_process(command, repo_root, runner_env, run_dir / "train.log")
        if return_code != 0:
            return return_code

        print(f"Run finished successfully: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
