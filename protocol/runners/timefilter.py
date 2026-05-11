from __future__ import annotations

from pathlib import Path

from protocol.manifests import Manifest
from protocol.runners.common import build_base_env, build_cli_args


CANONICAL_TIMEFILTER_ENTRY = Path("baselines/TimeFilter/run.py")


def build_command(repo_root: Path, manifest: Manifest, run_dir: Path):
    if manifest.repo_relative_entry != CANONICAL_TIMEFILTER_ENTRY:
        raise ValueError(
            "TimeFilter experiments must use the vendored protocol route "
            f"{CANONICAL_TIMEFILTER_ENTRY.as_posix()}, got {manifest.repo_relative_entry.as_posix()}."
        )
    if not isinstance(manifest.pred_len, int):
        raise TypeError("TimeFilter builder expects a single pred_len after manifest expansion.")

    args = dict(manifest.args)
    trials = manifest.search_budget.get("trials")
    if trials is not None:
        args.setdefault("itr", int(trials))

    args.pop("skip_predictions", None)
    args.setdefault("task_name", "long_term_forecast")
    args.setdefault("is_training", 1)
    args.setdefault("model", "TimeFilter")
    args.setdefault("model_id", f"{manifest.dataset.lower()}_{manifest.seq_len}_{manifest.pred_len}")
    args.setdefault("des", "protocol")
    args["pred_len"] = manifest.pred_len
    args["seed"] = manifest.seed
    args["run_id"] = run_dir.name
    args["output_dir"] = str(run_dir)
    args["metric_policy"] = manifest.metric_policy
    args["selection_policy"] = manifest.selection_policy
    args["checkpoints"] = str(run_dir / "checkpoints")
    root_path = args.get("root_path")
    if isinstance(root_path, str) and "$" not in root_path and not Path(root_path).is_absolute():
        args["root_path"] = str(repo_root / root_path)

    timefilter_root = repo_root / CANONICAL_TIMEFILTER_ENTRY.parent
    command = ["python", "-u", "run.py", *build_cli_args(args)]
    env = build_base_env(repo_root, manifest)
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = f"{timefilter_root}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = str(timefilter_root)
    return command, env, timefilter_root
