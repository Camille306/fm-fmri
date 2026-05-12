#!/usr/bin/env python3
"""
Collect fm-fmri test results from various possible locations and create a summary.
This script searches for test_results.txt files in common fm-fmri result locations.

Usage:
    python fm-fmri/collect_fm_results.py [--search_dir <directory>]
    
If --search_dir is not provided, searches in:
    - results/fm/<task>/
    - <any directory>/runs/<run_name>/ (ablation structure)
    - checkpoints_fmts/
    - Current directory and subdirectories
"""

import argparse
import re
import csv
from pathlib import Path
from collections import defaultdict


def parse_test_results(path: Path) -> dict:
    """Parse test_results.txt for MSE, MAE, FC similarity, and optional top-k metrics."""
    out = {"mse": None, "mae": None, "freq_diff": None, "fc_similarity": None, "num_subjects": None}
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8")
    for key, pattern in [
        ("mse", r"MSE.*?([0-9.e+-]+)\s*±"),
        ("mae", r"MAE.*?([0-9.e+-]+)\s*±"),
        ("freq_diff", r"(?:Freq diff|Frequency Difference|PSD MAE|PSD\s*\([^)]*\)).*?([0-9.e+-]+)\s*±"),
        ("fc_similarity", r"(?:FC sim(?:ilarity)?|Functional Connectivity Similarity).*?([0-9.e+-]+)\s*±"),
        ("num_subjects", r"Num subjects:\s*(\d+)"),
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                val = m.group(1).strip()
                if key == "num_subjects":
                    out[key] = int(val)
                else:
                    out[key] = float(val) if val.lower() != "nan" else 0.0
            except ValueError:
                pass
    # Optional top-k lines
    for k in (5, 10, 20, 50):
        m = re.search(
            rf"k={k}%\s*:\s*Precision\s+([0-9.e+-]+)\s*±[^\n]*Recall\s+([0-9.e+-]+)\s*±[^\n]*AUC\s+([0-9.e+-]+)",
            text,
        )
        if m:
            try:
                out[f"precision_at_{k}"] = float(m.group(1))
                out[f"recall_at_{k}"] = float(m.group(2))
                out[f"auc_at_{k}"] = float(m.group(3))
            except ValueError:
                pass
    return out


def find_fm_results(search_dirs):
    """Find all test_results.txt files that look like fm-fmri results."""
    results = []
    search_paths = [Path(d) for d in search_dirs if Path(d).exists()]
    
    # Also search current directory and repo root
    repo_root = Path(__file__).resolve().parent.parent
    search_paths.append(repo_root)
    search_paths.append(Path.cwd())
    
    # Common locations for fm-fmri results
    common_locations = [
        repo_root / "results" / "fm",
        repo_root / "checkpoints_fmts",
        Path.home() / "project" / "rest_to_task",  # Common SLURM output location
    ]
    for loc in common_locations:
        if loc.exists() and loc not in search_paths:
            search_paths.append(loc)
    
    seen_paths = set()  # Avoid duplicates
    
    for search_dir in search_paths:
        # Search for test_results.txt files
        for test_file in search_dir.rglob("test_results.txt"):
            if str(test_file) in seen_paths:
                continue
            seen_paths.add(str(test_file))
            
            # Check if it looks like an fm-fmri result (contains "FM-TS" or "Flow Matching")
            try:
                content = test_file.read_text(encoding="utf-8")
                if "FM-TS" in content or "Flow Matching" in content or "fm" in test_file.parent.name.lower():
                    metrics = parse_test_results(test_file)
                    if metrics.get("mse") is not None or metrics.get("fc_similarity") is not None:
                        # Try to infer task name from path
                        task = None
                        path_str = str(test_file)
                        for t in ["emotion", "gambling", "language", "motor", "relational", "social", "WM"]:
                            if t.lower() in path_str.lower():
                                task = t
                                break
                        
                        # Try to infer run/model name
                        run_name = test_file.parent.name
                        if test_file.parent.parent.name == "runs":
                            run_name = test_file.parent.name
                        elif "checkpoint" in path_str.lower():
                            run_name = "checkpoint"
                        
                        results.append({
                            "path": str(test_file),
                            "task": task,
                            "run_name": run_name,
                            **metrics
                        })
            except Exception as e:
                continue
    
    return results


def main():
    p = argparse.ArgumentParser(description="Collect fm-fmri test results from various locations")
    p.add_argument("--search_dir", type=str, action="append", default=[],
                   help="Directory to search for results (can be specified multiple times)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory for summary files (default: repo root/results)")
    args = p.parse_args()
    
    # Default search directories
    repo_root = Path(__file__).resolve().parent.parent
    default_search_dirs = [
        str(repo_root / "results" / "fm"),
        str(repo_root / "results"),
        str(repo_root),
    ]
    
    search_dirs = args.search_dir if args.search_dir else default_search_dirs
    
    print(f"Searching for fm-fmri results in: {search_dirs}")
    results = find_fm_results(search_dirs)
    
    if not results:
        print("No fm-fmri results found!")
        print("\nSearched in:")
        for d in search_dirs:
            print(f"  - {d}")
        print("\nCommon locations for fm-fmri results:")
        print("  - results/fm/<task>/test_results.txt")
        print("  - <output_base>/<task>/runs/<run_name>/test_results.txt")
        print("  - checkpoints_fmts/test_results.txt")
        print("\nIf your results are in a different location, use --search_dir to specify it.")
        print("\nExample:")
        print("  python fm-fmri/collect_fm_results.py --search_dir /path/to/your/results")
        return
    
    print(f"\nFound {len(results)} fm-fmri result files:")
    for r in results:
        print(f"  - {r['path']}")
    
    # Group by task and find best per task
    by_task = defaultdict(list)
    for r in results:
        task = r.get("task") or "unknown"
        by_task[task].append(r)
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = repo_root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write summary
    lines = ["FM-fMRI Test Results Summary", "=" * 70, ""]
    rows_for_csv = []
    
    for task in sorted(by_task.keys()):
        task_results = by_task[task]
        lines.append(f"Task: {task} ({len(task_results)} runs)")
        
        # Find best by FC similarity
        best_fc = None
        best_fc_val = -1
        for r in task_results:
            fc = r.get("fc_similarity")
            if fc is not None and fc > best_fc_val:
                best_fc_val = fc
                best_fc = r
        
        # Find best by MSE
        best_mse = None
        best_mse_val = float("inf")
        for r in task_results:
            mse = r.get("mse")
            if mse is not None and mse < best_mse_val:
                best_mse_val = mse
                best_mse = r
        
        if best_fc:
            lines.append(f"  Best FC similarity: {best_fc['run_name']} (FC={best_fc_val:.6f}, MSE={best_fc.get('mse', 'N/A')})")
            rows_for_csv.append({
                "task": task,
                "model": "fm",
                "run_name": best_fc["run_name"],
                "mse": best_fc.get("mse"),
                "mae": best_fc.get("mae"),
                "freq_diff": best_fc.get("freq_diff"),
                "fc_similarity": best_fc.get("fc_similarity"),
                "num_subjects": best_fc.get("num_subjects"),
                "path": best_fc["path"],
            })
        if best_mse and best_mse != best_fc:
            lines.append(f"  Best MSE: {best_mse['run_name']} (MSE={best_mse_val:.6f}, FC={best_mse.get('fc_similarity', 'N/A')})")
        lines.append("")
    
    # Write text summary
    summary_txt = output_dir / "fm_results_summary.txt"
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {summary_txt}")
    
    # Write CSV
    if rows_for_csv:
        csv_path = output_dir / "fm_results.csv"
        fieldnames = ["task", "model", "run_name", "mse", "mae", "freq_diff", "fc_similarity", "num_subjects", "path"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows_for_csv)
        print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
