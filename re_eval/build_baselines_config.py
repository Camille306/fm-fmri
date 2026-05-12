#!/usr/bin/env python3
"""
Build the baselines config JSON by scanning the cluster for best_fmts.pth checkpoints.

Looks under:
  {base_dir}/{experiment_name}/runs/*/best_fmts.pth

Default base_dir: ./
Default experiments: nhead8_ablation_emotion, nhead8_ev_all_tasks, nhead8_ablation_other_tasks

Each found path becomes an entry: load_dir = directory containing best_fmts.pth,
name = "{experiment_name}/{run_name}". task_name is set as follows:
- nhead8_ablation_emotion: always "emotion".
- nhead8_ev_all_tasks, nhead8_ablation_other_tasks: first word before first
  underscore in the run name (e.g. run "gambling_0" -> task_name "gambling").
- Other experiments: inferred from experiment name when possible.

Usage (on cluster):
    python re_eval/build_baselines_config.py
    python re_eval/build_baselines_config.py --base_dir ./ --out re_eval/baselines_config.json
"""

import argparse
import json
from pathlib import Path
from typing import Optional


# Experiment names to scan; optional task_name override when inferrable from name
DEFAULT_EXPERIMENTS = [
    "nhead8_ablation_emotion",
    "nhead8_ev_all_tasks",
    "nhead8_ablation_other_tasks",
]


# For these experiments, task_name comes from the run name (first segment before first _)
TASK_FROM_RUN_NAME = ("nhead8_ev_all_tasks", "nhead8_ablation_other_tasks")


def run_name_to_task(run_name: str) -> Optional[str]:
    """First word before the first underscore in run name (e.g. gambling_0 -> gambling)."""
    if not run_name:
        return None
    parts = run_name.split("_", 1)
    return parts[0] if parts[0] else None


def experiment_to_task(experiment_name: str) -> Optional[str]:
    """If we can infer a single task from experiment name, return it; else None."""
    if "emotion" in experiment_name and "other" not in experiment_name.lower():
        return "emotion"
    if "gambling" in experiment_name:
        return "gambling"
    if "language" in experiment_name:
        return "language"
    if "motor" in experiment_name:
        return "motor"
    if "relational" in experiment_name:
        return "relational"
    if "social" in experiment_name:
        return "social"
    return None


def main():
    p = argparse.ArgumentParser(description="Build baselines config JSON from cluster checkpoint paths")
    p.add_argument(
        "--base_dir",
        type=str,
        default="./",
        help="Base directory containing experiment folders",
    )
    p.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=DEFAULT_EXPERIMENTS,
        help="Experiment names to scan (each: base_dir/NAME/runs/*/best_fmts.pth)",
    )
    p.add_argument(
        "--out",
        type=str,
        default="re_eval/baselines_config.json",
        help="Output JSON config path",
    )
    p.add_argument(
        "--no_task_name",
        action="store_true",
        help="Do not add task_name to any entry (rely on CLI --task_name)",
    )
    args = p.parse_args()

    base = Path(args.base_dir)
    if not base.is_dir():
        print(f"Warning: base_dir does not exist: {base}")
        print("Will still write config for any experiments that exist.")

    config = []
    for exp in args.experiments:
        runs_dir = base / exp / "runs"
        if not runs_dir.is_dir():
            print(f"Skip (no runs dir): {runs_dir}")
            continue
        n_runs = 0
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            best_pth = run_dir / "best_fmts.pth"
            if not best_pth.is_file():
                continue
            if args.no_task_name:
                task_name = None
            elif exp in TASK_FROM_RUN_NAME:
                task_name = run_name_to_task(run_dir.name)
            else:
                task_name = experiment_to_task(exp)
            load_dir = str(run_dir.resolve())
            name = f"{exp}/{run_dir.name}"
            entry = {"name": name, "load_dir": load_dir}
            if task_name is not None:
                entry["task_name"] = task_name
            config.append(entry)
            n_runs += 1
        print(f"  {exp}: {n_runs} runs")

    # Re-count per experiment for print
    total = len(config)
    print(f"Total entries: {total}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
