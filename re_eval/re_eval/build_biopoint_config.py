"""Build re_eval config for Biopoint runs (FM-TS and baselines).

Same layout as biopoint/slurm and biopoint/slurm/collect_* scripts:
  - FM: results_flow_matching_biopoint (single) or output_base/runs/<run_name>/ (sweep/ablations)
  - Baselines: results_biopoint/<model>/  (timegan, timevae, diffusion_ts, ddpm)

Run from this directory (re_eval/re_eval/):
  python build_biopoint_config.py
  python build_biopoint_config.py --fm_base /path/to/results_flow_matching_biopoint
  python build_biopoint_config.py --sweep_base /path/to/results_fm_biopoint_sweep_freq_fc
"""
import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FM_BASE = REPO_ROOT / "results_flow_matching_biopoint"
DEFAULT_BIOPOINT_RESULTS = REPO_ROOT / "results_biopoint"
DEFAULT_SWEEP_BASE = REPO_ROOT / "results_fm_biopoint_sweep_freq_fc"
DEFAULT_ABLATIONS_BASE = REPO_ROOT / "results_flow_matching_biopoint_ablations"

BASELINE_MODELS = ["timegan", "timevae", "diffusion_ts", "ddpm", "lstm_gan"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fm_base", type=str, default=None,
                   help=f"Single FM run dir (contains best_fmts.pth). Default: {DEFAULT_FM_BASE}")
    p.add_argument("--sweep_base", type=str, default=None,
                   help=f"Sweep output base (sweep_base/runs/<name>/). Default: {DEFAULT_SWEEP_BASE}")
    p.add_argument("--ablations_base", type=str, default=None,
                   help=f"Ablations output base (ablations_base/runs/<name>/). Default: {DEFAULT_ABLATIONS_BASE}")
    p.add_argument("--results_biopoint", type=str, default=None,
                   help=f"Biopoint baselines root (results_biopoint/<model>/). Default: {DEFAULT_BIOPOINT_RESULTS}")
    p.add_argument("--no_fm", action="store_true", help="Omit FM entries")
    p.add_argument("--no_baselines", action="store_true", help="Omit baseline entries")
    args = p.parse_args()

    config = []
    fm_base = Path(args.fm_base or DEFAULT_FM_BASE).resolve() if not args.no_fm else None
    sweep_base = Path(args.sweep_base or DEFAULT_SWEEP_BASE).resolve() if not args.no_fm else None
    ablations_base = Path(args.ablations_base or DEFAULT_ABLATIONS_BASE).resolve() if not args.no_fm else None
    results_biopoint = Path(args.results_biopoint or DEFAULT_BIOPOINT_RESULTS).resolve() if not args.no_baselines else None

    # FM-TS entries
    if not args.no_fm:
        if fm_base and (fm_base / "best_fmts.pth").exists():
            config.append({
                "name": "biopoint_fm/single",
                "load_dir": str(fm_base),
                "model_type": "fmts",
                "data_source": "biopoint",
            })
        for base, label in [(sweep_base, "sweep"), (ablations_base, "ablations")]:
            if not base:
                continue
            runs_dir = base / "runs"
            if runs_dir.exists():
                for d in sorted(runs_dir.iterdir()):
                    if d.is_dir() and (d / "best_fmts.pth").exists():
                        config.append({
                            "name": f"biopoint_fm/{label}/{d.name}",
                            "load_dir": str(d),
                            "model_type": "fmts",
                            "data_source": "biopoint",
                        })

    # Baseline entries (same layout as biopoint/slurm/run_baseline.sh and collect_baselines_biopoint.py)
    if not args.no_baselines and results_biopoint:
        for m in BASELINE_MODELS:
            load_dir = results_biopoint / m
            config.append({
                "name": f"biopoint_baseline/{m}",
                "load_dir": str(load_dir),
                "model_type": m,
                "data_source": "biopoint",
            })

    with open("biopoint.json", "w") as f:
        json.dump(config, f, indent=2)
    print("Biopoint config entries:", len(config))
    print("  FM-TS:", sum(1 for e in config if e.get("model_type") == "fmts"))
    print("  Baselines:", sum(1 for e in config if e.get("model_type") in BASELINE_MODELS))
    print("Output: biopoint.json")


if __name__ == "__main__":
    main()
