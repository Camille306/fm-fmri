#!/usr/bin/env python3
"""
Collect test_results.txt from all model submission runs into a summary CSV
and print the best config per task.

Usage:
  python model_submission/collect_results.py
  python model_submission/collect_results.py --output_base /path/to/model_submission
  python model_submission/collect_results.py --best_by mse
"""
import argparse
import csv
import re
from pathlib import Path


def parse_test_results(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8")
    m = re.search(r"MSE \(mean [±+] std\):\s*([\d.]+)\s*[±+]\s*([\d.]+)", text)
    if m:
        out["mse"] = float(m.group(1))
        out["mse_std"] = float(m.group(2))
    m = re.search(r"MAE \(mean [±+] std\):\s*([\d.]+)\s*[±+]\s*([\d.]+)", text)
    if m:
        out["mae"] = float(m.group(1))
        out["mae_std"] = float(m.group(2))
    m = re.search(r"PSD.*\(mean [±+] std\):\s*([\d.]+)\s*[±+]\s*([\d.]+)", text)
    if m:
        out["psd"] = float(m.group(1))
        out["psd_std"] = float(m.group(2))
    m = re.search(r"FC sim \(mean [±+] std\):\s*([\d.]+)\s*[±+]\s*([\d.]+)", text)
    if m:
        out["fc_sim"] = float(m.group(1))
        out["fc_sim_std"] = float(m.group(2))
    m = re.search(r"Num subjects:\s*(\d+)", text)
    if m:
        out["num_subjects"] = int(m.group(1))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_base", type=str, default=".//model_submission")
    p.add_argument("--best_by", type=str, default="fc_sim", choices=("fc_sim", "mse"))
    args = p.parse_args()

    base = Path(args.output_base)
    runs_dir = base / "runs"

    if not runs_dir.exists():
        print(f"No runs dir at {runs_dir}")
        return

    rows = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics = parse_test_results(run_dir / "test_results.txt")
        if not metrics:
            continue
        # Extract task from run name (first token before _)
        name = run_dir.name
        task = name.split("_")[0]
        rows.append({"run": name, "task": task, "path": str(run_dir), **metrics})

    if not rows:
        print("No test_results.txt found in any run.")
        return

    # Write CSV
    csv_path = base / "all_results.csv"
    fieldnames = ["run", "task", "mse", "mse_std", "mae", "mae_std", "psd", "psd_std", "fc_sim", "fc_sim_std", "num_subjects", "path"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path} ({len(rows)} runs)")

    # Best per task
    tasks = sorted(set(r["task"] for r in rows))
    lower = args.best_by == "mse"

    print(f"\n{'='*100}")
    print(f"Best config per task (by {args.best_by})")
    print(f"{'='*100}")
    fmt = "{:<12} {:<40} {:>10} {:>10} {:>10} {:>10}"
    print(fmt.format("Task", "Config", "MSE", "MAE", "PSD", "FC sim"))
    print("-" * 100)

    best_rows = []
    for task in tasks:
        task_rows = [r for r in rows if r["task"] == task and isinstance(r.get(args.best_by), (int, float))]
        if not task_rows:
            continue
        if lower:
            best = min(task_rows, key=lambda r: r[args.best_by])
        else:
            best = max(task_rows, key=lambda r: r[args.best_by])
        config = best["run"].replace(f"{task}_", "", 1)
        print(fmt.format(
            task,
            config[:39],
            f"{best.get('mse', 0):.6f}",
            f"{best.get('mae', 0):.6f}",
            f"{best.get('psd', 0):.6f}",
            f"{best.get('fc_sim', 0):.6f}",
        ))
        best_rows.append({"task": task, "best_config": config, "path": best["path"],
                          "mse": best.get("mse", ""), "fc_sim": best.get("fc_sim", "")})

    print(f"{'='*100}")

    # Best per task CSV
    best_csv = base / "best_per_task.csv"
    with open(best_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task", "best_config", "mse", "fc_sim", "path"])
        w.writeheader()
        w.writerows(best_rows)
    print(f"\nBest per task written to {best_csv}")

    # All results txt (raw dump)
    txt_path = base / "all_results.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            res_path = run_dir / "test_results.txt"
            f.write(f"=== {run_dir.name} ===\n")
            if res_path.exists():
                f.write(res_path.read_text(encoding="utf-8"))
            else:
                f.write("(no results)\n")
            f.write("\n")
    print(f"Raw results dump written to {txt_path}")


if __name__ == "__main__":
    main()
