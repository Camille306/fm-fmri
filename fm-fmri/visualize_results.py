#!/usr/bin/env python3
"""
Visualize fm-fmri ablation results from CSV files or result directories.

This script can:
1. Read from ablation_results.csv (from collect_and_visualize_ablation.py)
2. Read from fm_results.csv (from collect_fm_results.py)
3. Read directly from result directories
4. Create bar charts, comparison plots, and summary visualizations

Usage:
    # From CSV file
    python fm-fmri/visualize_results.py --csv results/fm_results.csv
    
    # From ablation results directory
    python fm-fmri/visualize_results.py --output_base /path/to/ablation/results
    
    # Custom output directory
    python fm-fmri/visualize_results.py --csv results/fm_results.csv --output_dir results/figures
"""

import argparse
import csv
import re
from pathlib import Path
from collections import defaultdict
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


def parse_test_results(path: Path) -> dict:
    """Parse test_results.txt for metrics."""
    out = {}
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8")
    patterns = [
        (r"MSE \(mean ± std\):\s*([\d.]+)\s*±\s*([\d.]+)", "mse", "mse_std"),
        (r"MAE \(mean ± std\):\s*([\d.]+)\s*±\s*([\d.]+)", "mae", "mae_std"),
        (r"PSD MAE \(mean ± std\):\s*([\d.]+)\s*±\s*([\d.]+)", "freq_diff", "freq_diff_std"),
        (r"FC sim \(mean ± std\):\s*([\d.]+)\s*±\s*([\d.]+)", "fc_similarity", "fc_similarity_std"),
        (r"Num subjects:\s*(\d+)", "num_subjects", None),
    ]
    for pattern, key, std_key in patterns:
        m = re.search(pattern, text)
        if m:
            out[key] = float(m.group(1)) if key != "num_subjects" else int(m.group(1))
            if std_key and len(m.groups()) > 1:
                out[std_key] = float(m.group(2))
    return out


def load_from_csv(csv_path: Path):
    """Load results from CSV file."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert string values to numbers
            for key in ["mse", "mae", "freq_diff", "fc_similarity", "num_subjects",
                       "mse_std", "mae_std", "freq_diff_std", "fc_similarity_std"]:
                if key in row and row[key]:
                    try:
                        row[key] = float(row[key]) if key != "num_subjects" else int(row[key])
                    except (ValueError, TypeError):
                        row[key] = None
            rows.append(row)
    return rows


def load_from_directory(output_base: Path):
    """Load results from directory structure (runs/ subdirectory)."""
    runs_dir = output_base / "runs"
    if not runs_dir.exists():
        return []
    
    rows = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        test_file = run_dir / "test_results.txt"
        if not test_file.exists():
            continue
        
        metrics = parse_test_results(test_file)
        if metrics:
            metrics["run"] = run_dir.name
            metrics["run_path"] = str(run_dir.resolve())
            rows.append(metrics)
    return rows


def bar_plot(run_names, values, yerr, title, ylabel, fpath, figsize=None):
    """Create a bar plot with error bars."""
    if not HAS_MATPLOTLIB:
        print(f"Skipping plot {fpath}: matplotlib not available")
        return
    
    x = np.arange(len(run_names))
    if figsize is None:
        figsize = (max(6, len(run_names) * 0.6), 5)
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, values, yerr=yerr, capsize=4, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(run_names, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fpath}")


def comparison_plot(rows, output_dir: Path):
    """Create comparison plots across different metrics."""
    if not rows:
        return
    
    # Extract data
    run_names = [r.get("run", r.get("run_name", f"run_{i}")) for i, r in enumerate(rows)]
    metrics_data = {
        "MSE": ([r.get("mse") for r in rows], [r.get("mse_std", 0) for r in rows]),
        "MAE": ([r.get("mae") for r in rows], [r.get("mae_std", 0) for r in rows]),
        "FC Similarity": ([r.get("fc_similarity") for r in rows], [r.get("fc_similarity_std", 0) for r in rows]),
        "PSD MAE": ([r.get("freq_diff") for r in rows], [r.get("freq_diff_std", 0) for r in rows]),
    }
    
    # Create individual plots
    for metric_name, (values, stds) in metrics_data.items():
        valid_indices = [i for i, v in enumerate(values) if v is not None]
        if not valid_indices:
            continue
        
        valid_names = [run_names[i] for i in valid_indices]
        valid_values = [values[i] for i in valid_indices]
        valid_stds = [stds[i] for i in valid_indices]
        
        fpath = output_dir / f"comparison_{metric_name.lower().replace(' ', '_')}.pdf"
        bar_plot(valid_names, valid_values, valid_stds, 
                f"FM-fMRI: {metric_name} Comparison", metric_name, fpath)


def grouped_ablation_plots(rows, output_dir: Path):
    """Create grouped plots for different ablation categories."""
    if not rows:
        return
    
    # Group by prefix
    groups = defaultdict(list)
    for r in rows:
        run_name = r.get("run", r.get("run_name", ""))
        if run_name.startswith("ablation_loss_"):
            groups["loss_ablation"].append(r)
        elif run_name.startswith("ablation_ev_"):
            groups["ev_ablation"].append(r)
        elif run_name.startswith("tune_ev_"):
            groups["hyperparameter_tuning"].append(r)
        elif run_name.startswith("ablation_rest_encoder_"):
            groups["rest_encoder"].append(r)
        elif run_name.startswith("ablation_prior_detach_"):
            groups["prior_detach"].append(r)
    
    # Create plots for each group
    for group_name, group_rows in groups.items():
        if not group_rows:
            continue
        
        # Clean names
        run_names = [r.get("run", r.get("run_name", "")) for r in group_rows]
        clean_names = [name.split("_", 2)[-1] if "_" in name else name for name in run_names]
        
        # MSE plot
        mses = [r.get("mse") for r in group_rows]
        mse_stds = [r.get("mse_std", 0) for r in group_rows]
        valid = [i for i, v in enumerate(mses) if v is not None]
        if valid:
            bar_plot([clean_names[i] for i in valid],
                    [mses[i] for i in valid],
                    [mse_stds[i] for i in valid],
                    f"Ablation: {group_name.replace('_', ' ').title()}", "MSE",
                    output_dir / f"{group_name}_mse.pdf")
        
        # FC similarity plot
        fcs = [r.get("fc_similarity") for r in group_rows]
        fc_stds = [r.get("fc_similarity_std", 0) for r in group_rows]
        valid = [i for i, v in enumerate(fcs) if v is not None]
        if valid:
            bar_plot([clean_names[i] for i in valid],
                    [fcs[i] for i in valid],
                    [fc_stds[i] for i in valid],
                    f"Ablation: {group_name.replace('_', ' ').title()}", "FC Similarity",
                    output_dir / f"{group_name}_fc.pdf")


def task_comparison_plot(rows, output_dir: Path):
    """Create plots comparing results across tasks."""
    if not rows:
        return
    
    # Group by task
    by_task = defaultdict(list)
    for r in rows:
        task = r.get("task", "unknown")
        by_task[task].append(r)
    
    if len(by_task) < 2:
        return  # Need at least 2 tasks to compare
    
    # Find best per task (by FC similarity)
    best_per_task = {}
    for task, task_rows in by_task.items():
        best = None
        best_fc = -1
        for r in task_rows:
            fc = r.get("fc_similarity")
            if fc is not None and fc > best_fc:
                best_fc = fc
                best = r
        if best:
            best_per_task[task] = best
    
    if len(best_per_task) < 2:
        return
    
    # Create comparison plot
    tasks = sorted(best_per_task.keys())
    metrics = {
        "MSE": [best_per_task[t].get("mse") for t in tasks],
        "FC Similarity": [best_per_task[t].get("fc_similarity") for t in tasks],
    }
    
    for metric_name, values in metrics.items():
        valid = [(i, v) for i, v in enumerate(values) if v is not None]
        if not valid:
            continue
        
        valid_indices, valid_values = zip(*valid)
        valid_tasks = [tasks[i] for i in valid_indices]
        
        if HAS_MATPLOTLIB:
            fig, ax = plt.subplots(figsize=(max(6, len(valid_tasks) * 0.8), 5))
            ax.bar(valid_tasks, valid_values, alpha=0.7)
            ax.set_ylabel(metric_name)
            ax.set_title(f"FM-fMRI: {metric_name} by Task")
            ax.grid(axis="y", alpha=0.3)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            fname = f"task_comparison_{metric_name.lower().replace(' ', '_')}.pdf"
            plt.savefig(output_dir / fname,
                       dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved {output_dir / fname}")


def main():
    p = argparse.ArgumentParser(description="Visualize fm-fmri ablation results")
    p.add_argument("--csv", type=str, default=None, help="Path to CSV file with results")
    p.add_argument("--output_base", type=str, default=None, 
                   help="Path to output base directory (looks for runs/ subdirectory)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory for figures (default: figures/ in same dir as input)")
    args = p.parse_args()
    
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required for visualization.")
        print("Install with: pip install matplotlib")
        sys.exit(1)
    
    # Load data
    rows = []
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"Error: CSV file not found: {csv_path}")
            sys.exit(1)
        rows = load_from_csv(csv_path)
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = csv_path.parent / "figures"
    elif args.output_base:
        output_base = Path(args.output_base)
        if not output_base.exists():
            print(f"Error: Output base directory not found: {output_base}")
            sys.exit(1)
        rows = load_from_directory(output_base)
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = output_base / "figures"
    else:
        print("Error: Must specify either --csv or --output_base")
        p.print_help()
        sys.exit(1)
    
    if not rows:
        print("No results found!")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(rows)} results")
    print(f"Output directory: {output_dir}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    comparison_plot(rows, output_dir)
    grouped_ablation_plots(rows, output_dir)
    task_comparison_plot(rows, output_dir)
    
    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
