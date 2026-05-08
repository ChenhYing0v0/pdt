from __future__ import annotations

from pathlib import Path

from protocol.manifests import Manifest
from protocol.runners.common import build_base_env, build_cli_args


CANONICAL_DLINEAR_ENTRY = Path("baselines/PDT/run.py")


def build_command(repo_root: Path, manifest: Manifest, run_dir: Path) -> tuple[list[str], dict[str, str]]:
    args = dict(manifest.args)
    if not isinstance(manifest.pred_len, int):
        raise TypeError("DLinear builder expects a single pred_len after manifest expansion.")
    if manifest.repo_relative_entry != CANONICAL_DLINEAR_ENTRY:
        raise ValueError(
            "DLinear experiments must use the verified shared PDT model-zoo route "
            f"{CANONICAL_DLINEAR_ENTRY.as_posix()}, got {manifest.repo_relative_entry.as_posix()}."
        )

    trials = manifest.search_budget.get("trials")
    if trials is not None:
        args.setdefault("itr", int(trials))

    args.setdefault("task_name", "long_term_forecast")
    args.setdefault("is_training", 1)
    args.setdefault("model", "DLinear")
    args.setdefault("model_id", f"{manifest.dataset.lower()}_pl{manifest.pred_len}")
    args.setdefault("des", "R3_1Clean")
    args.setdefault("loss_mode", "L2")
    args["pred_len"] = manifest.pred_len
    args["seed"] = manifest.seed
    args["run_id"] = run_dir.name
    args["output_dir"] = str(run_dir)
    args["metric_policy"] = manifest.metric_policy
    args["selection_policy"] = manifest.selection_policy

    entry = repo_root / CANONICAL_DLINEAR_ENTRY
    command = ["python", "-u", str(entry), *build_cli_args(args)]
    env = build_base_env(repo_root, manifest)
    return command, env
