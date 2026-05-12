"""Build fm_fmri_and_baseline.json or baseline-only config from fm_fmri.json + baseline dirs.

Baseline load_dir uses the same layout as baselines/slurm and baselines/slurm/collect_results.py:
  results/<model>/<task>/   (under repo root, with best_*.pth and test_results.txt).

Run from this directory (re_eval/re_eval/):
  python build_combined_config.py              # full combined config
  python build_combined_config.py --baseline_only   # baseline entries only -> fm_baseline_only.json
  python build_combined_config.py --baseline_only --results_base /path/to/results   # custom results root
"""
import argparse
import json
from pathlib import Path

# Same convention as baselines/slurm/run_baseline.sh and baselines/slurm/collect_results.py
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RESULTS_BASE = REPO_ROOT / "results"

MODELS = ["timegan", "timevae", "diffusion_ts", "ddpm", "lstm_gan"]
TASKS = ["emotion", "gambling", "language", "motor", "relational", "social", "WM"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_only", action="store_true", help="Write only baseline entries to fm_baseline_only.json")
    p.add_argument("--results_base", type=str, default=None,
                   help=f"Directory containing <model>/<task>/ checkpoints (default: repo root / results = {DEFAULT_RESULTS_BASE})")
    args = p.parse_args()

    results_base = Path(args.results_base) if args.results_base else DEFAULT_RESULTS_BASE
    results_base = results_base.resolve()

    if args.baseline_only:
        config = []
        for m in MODELS:
            for t in TASKS:
                config.append({
                    "name": f"baseline/{m}/{t}",
                    "load_dir": str(results_base / m / t),
                    "task_name": t,
                    "model_type": m,
                })
        out_path = "fm_baseline_only.json"
    else:
        with open("fm_fmri.json") as f:
            config = json.load(f)
        for e in config:
            e["model_type"] = e.get("model_type", "fmts")
        for m in MODELS:
            for t in TASKS:
                config.append({
                    "name": f"baseline/{m}/{t}",
                    "load_dir": str(results_base / m / t),
                    "task_name": t,
                    "model_type": m,
                })
        out_path = "fm_fmri_and_baseline.json"

    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    print("Entries:", len(config))
    print("Baseline load_dir base:", results_base)
    if args.baseline_only:
        print("Output:", out_path)
    else:
        print("FM-TS:", sum(1 for e in config if e.get("model_type") == "fmts"))
        print("Baselines:", sum(1 for e in config if e.get("model_type") != "fmts"))
        print("Output:", out_path)


if __name__ == "__main__":
    main()
