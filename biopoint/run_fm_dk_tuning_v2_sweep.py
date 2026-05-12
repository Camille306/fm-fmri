#!/usr/bin/env python3
"""
Run the FM Biopoint DK tuning v2 parameter sweep locally (no SLURM).

Iterates over the same 96-config grid as run_fm_biopoint_dk_tuning_v2.slurm.

Usage (from repo root):
  python biopoint/run_fm_dk_tuning_v2_sweep.py                  # run all 96
  python biopoint/run_fm_dk_tuning_v2_sweep.py --job_ids 0 1 5   # run specific indices
  python biopoint/run_fm_dk_tuning_v2_sweep.py --job_ids 0-11    # run a range
"""

import argparse
import itertools
import os
import subprocess
import sys
from pathlib import Path


GRID = {
    "lr":   ["1e-3", "5e-4"],
    "fc":   ["0.3", "0.5", "1.0", "2.0"],
    "freq": ["0.1", "0.3", "0.5"],
    "coh":  ["0.0", "0.1"],
    "d_ev": ["32", "64"],
}

FIXED = {
    "hidden": "128",
    "lookback": "100",
    "epochs": "100",
    "batch_size": "8",
    "prediction_length": "46",
    "stride": "5",
    "t_dim": "128",
    "num_conditions": "32",
    "aux_ode_steps": "10",
    "ode_steps": "50",
    "max_grad_norm": "1.0",
    "num_workers": "1",
}


def build_configs():
    """Generate all grid configs in the same order as the SLURM array."""
    keys = ["lr", "fc", "freq", "coh", "d_ev"]
    value_lists = [GRID[k] for k in keys]
    configs = []
    for idx, combo in enumerate(itertools.product(*value_lists)):
        cfg = dict(zip(keys, combo))
        cfg["idx"] = idx
        configs.append(cfg)
    return configs


def make_tag(cfg):
    tag = f"lr{cfg['lr']}_fc{cfg['fc']}_freq{cfg['freq']}_coh{cfg['coh']}_dev{cfg['d_ev']}_h{FIXED['hidden']}_lb{FIXED['lookback']}"
    return tag.replace(".", "p")


def parse_job_ids(raw_ids, n_total):
    """Parse job ID specs like '0', '5', '0-11' into a set of ints."""
    result = set()
    for spec in raw_ids:
        if "-" in spec:
            lo, hi = spec.split("-", 1)
            result.update(range(int(lo), int(hi) + 1))
        else:
            result.add(int(spec))
    return sorted(i for i in result if 0 <= i < n_total)


def main():
    parser = argparse.ArgumentParser(description="FM DK tuning v2 sweep (local, no SLURM)")
    parser.add_argument("--job_ids", nargs="*", default=None,
                        help="Subset of job indices to run (e.g. 0 1 5 or 0-11). Default: all 96.")
    parser.add_argument("--data_root", default="./data/biopoint_data")
    parser.add_argument("--csv_path", default="./data/biopoint_data.csv")
    parser.add_argument("--eprime_root", default="~/project_pi/user/rest_to_task/biopoint/eprime_biopoint")
    parser.add_argument("--dk_atlas_ts_root", default="./data/biopoint_dk_atlas")
    parser.add_argument("--output_base", default=None,
                        help="Output directory (default: <repo>/results_fm_biopoint_dk_tuning_v2)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip configs where best_fmts.pth already exists")
    parser.add_argument("--device", default=None, help="Force device (default: auto-detect)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "biopoint" / "run_flow_matching_biopoint.py"
    output_base = Path(args.output_base) if args.output_base else (repo_root / "results_fm_biopoint_dk_tuning_v2")
    output_base.mkdir(parents=True, exist_ok=True)

    eprime_root = os.path.expanduser(args.eprime_root)

    configs = build_configs()
    if args.job_ids:
        selected = parse_job_ids(args.job_ids, len(configs))
        configs = [configs[i] for i in selected]
    n_total = len(configs)

    print(f"FM DK tuning v2: {n_total} configs to run")
    print(f"Output: {output_base}\n")

    for run_i, cfg in enumerate(configs):
        idx = cfg["idx"]
        tag = make_tag(cfg)
        save_dir = output_base / f"{idx}_{tag}"
        save_dir.mkdir(parents=True, exist_ok=True)

        if args.skip_existing and (save_dir / "best_fmts.pth").is_file():
            print(f"[{run_i+1}/{n_total}] Skip idx={idx} (checkpoint exists): {tag}")
            continue

        print(f"\n{'='*70}")
        print(f"[{run_i+1}/{n_total}] idx={idx}  lr={cfg['lr']}  fc={cfg['fc']}  "
              f"freq={cfg['freq']}  coh={cfg['coh']}  d_ev={cfg['d_ev']}")
        print(f"{'='*70}")

        cmd = [
            sys.executable, str(script),
            "--data_root", args.data_root,
            "--csv_path", args.csv_path,
            "--eprime_root", eprime_root,
            "--atlas_source", "dk",
            "--dk_atlas_ts_root", args.dk_atlas_ts_root,
            "--save_dir", str(save_dir),
            "--epochs", FIXED["epochs"],
            "--batch_size", FIXED["batch_size"],
            "--lr", cfg["lr"],
            "--lookback_length", FIXED["lookback"],
            "--prediction_length", FIXED["prediction_length"],
            "--stride", FIXED["stride"],
            "--rest_hidden", FIXED["hidden"],
            "--ctx_dim", FIXED["hidden"],
            "--t_dim", FIXED["t_dim"],
            "--d_ev", cfg["d_ev"],
            "--num_conditions", FIXED["num_conditions"],
            "--freq_loss_weight", cfg["freq"],
            "--fc_loss_weight", cfg["fc"],
            "--coh_loss_weight", cfg["coh"],
            "--aux_ode_steps", FIXED["aux_ode_steps"],
            "--ode_steps", FIXED["ode_steps"],
            "--max_grad_norm", FIXED["max_grad_norm"],
            "--num_workers", FIXED["num_workers"],
        ]
        if args.device:
            cmd += ["--device", args.device]

        print("  $ " + " ".join(cmd))
        result = subprocess.run(cmd, cwd=str(repo_root))
        if result.returncode != 0:
            print(f"  WARNING: idx={idx} exited with code {result.returncode}")

    print(f"\n{'='*70}")
    print(f"Sweep complete. Collect results with:")
    print(f"  python biopoint/slurm/collect_fm_biopoint_param_search.py \\")
    print(f"    --results_dir {output_base} --top_n 10")


if __name__ == "__main__":
    main()
