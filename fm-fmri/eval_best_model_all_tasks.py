#!/usr/bin/env python3
"""
Evaluate the best performing fm-fmri model on all tasks.

This script:
1. Optionally finds the best model from ablation results
2. Evaluates the best model on all 7 tasks
3. Saves results to results/fm/best_model_<task>/
4. Collects and summarizes all results

Usage:
    # Option 1: Specify best model path directly
    python fm-fmri/eval_best_model_all_tasks.py \
        --load_dir /path/to/best/model \
        --data_root /path/to/hcp-resting-fc \
        --task_root /path/to/hcp-task-ts

    # Option 2: Auto-find best model from ablation results
    python fm-fmri/eval_best_model_all_tasks.py \
        --ablation_base /path/to/ablation/results \
        --data_root /path/to/hcp-resting-fc \
        --task_root /path/to/hcp-task-ts \
        --best_by fc_similarity
"""

import argparse
import subprocess
import sys
from pathlib import Path
import os


def find_best_model(ablation_base: Path, best_by: str = "fc_similarity") -> Path:
    """Find best model path from ablation results."""
    print(f"Finding best model from ablation results...")
    print(f"  Ablation base: {ablation_base}")
    print(f"  Best by: {best_by}")
    
    # Run collect_and_visualize_ablation.py
    script_path = Path(__file__).parent / "slurm" / "collect_and_visualize_ablation.py"
    result = subprocess.run(
        [sys.executable, str(script_path), 
         "--output_base", str(ablation_base),
         "--best_by", best_by],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error: Failed to find best model")
        print(result.stderr)
        sys.exit(1)
    
    best_model_path_file = ablation_base / "best_model_path.txt"
    if not best_model_path_file.exists():
        print(f"Error: best_model_path.txt not found at {best_model_path_file}")
        print("Check that ablation_results.csv exists and contains valid results.")
        sys.exit(1)
    
    load_dir = Path(best_model_path_file.read_text().strip())
    print(f"  Found best model: {load_dir}")
    return load_dir


def validate_model_dir(load_dir: Path) -> bool:
    """Validate that model directory exists and contains checkpoint."""
    if not load_dir.exists():
        print(f"Error: Model directory not found: {load_dir}")
        return False
    
    checkpoint = load_dir / "best_fmts.pth"
    if not checkpoint.exists():
        # Look for any .pth file
        pth_files = list(load_dir.glob("*.pth"))
        if not pth_files:
            print(f"Error: No checkpoint file found in {load_dir}")
            return False
        print(f"Warning: best_fmts.pth not found, using {pth_files[0]}")
    
    return True


def run_evaluation(
    load_dir: Path,
    task: str,
    data_root: Path,
    task_root: Path,
    save_dir: Path,
    use_evs: bool = False,
    ev_root: Path = None,
) -> bool:
    """Run evaluation for a single task."""
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "fm_fmri.py"),
        "--load_dir", str(load_dir),
        "--task_name", task,
        "--data_root", str(data_root),
        "--task_root", str(task_root),
        "--save_dir", str(save_dir),
    ]
    
    if use_evs:
        cmd.append("--use_evs")
    
    if ev_root:
        cmd.extend(["--ev_root", str(ev_root)])
    
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"  ✓ Success")
        return True
    else:
        print(f"  ✗ Failed")
        print(f"  Error output: {result.stderr[:500]}")
        return False


def main():
    p = argparse.ArgumentParser(
        description="Evaluate best fm-fmri model on all tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Direct model path
  python %(prog)s --load_dir ./checkpoints_fmts \\
      --data_root /data/hcp --task_root /data/task

  # Auto-find best model
  python %(prog)s --ablation_base ./ablation_results \\
      --best_by fc_similarity
        """
    )
    
    # Model selection
    p.add_argument("--load_dir", type=Path, default=None, 
                   help="Direct path to best model checkpoint directory (if not specified, uses --ablation_base)")
    p.add_argument("--ablation_base", type=Path, 
                   default=Path("./"),
                   help="Path to ablation results (auto-find best model) (default: ./)")
    
    # Data paths
    p.add_argument("--data_root", type=Path, 
                   default=os.getenv("DATA_ROOT", "./data/hcp-resting-fc"),
                   help="Path to HCP resting-state data (default: ./data/hcp-resting-fc)")
    p.add_argument("--task_root", type=Path,
                   default=os.getenv("TASK_ROOT", "./data/hcp-task-ts"),
                   help="Path to HCP task data (default: ./data/hcp-task-ts)")
    p.add_argument("--results_dir", type=Path, default=Path("results"),
                   help="Where to save evaluation results (default: ./results)")
    
    # Best model selection
    p.add_argument("--best_by", type=str, default="fc_similarity",
                   choices=("fc_similarity", "mse"),
                   help="Metric to use for best model selection (default: fc_similarity)")
    
    # Event files
    p.add_argument("--use_evs", action="store_true",
                   help="Use event files if model was trained with them")
    p.add_argument("--ev_root", type=Path, default=None,
                   help="Path to event files root directory (default: auto-detect)")
    
    # Options
    p.add_argument("--skip_collect", action="store_true",
                   help="Skip collecting results at the end")
    
    args = p.parse_args()
    
    # Validate that at least one model source is provided
    if args.load_dir is None and args.ablation_base is None:
        p.error("Must specify either --load_dir or --ablation_base")
    
    # Find best model if needed
    if args.load_dir is None:
        if not args.ablation_base.exists():
            print(f"Error: Ablation base directory not found: {args.ablation_base}")
            print("Please specify --load_dir or ensure --ablation_base points to valid directory")
            sys.exit(1)
        load_dir = find_best_model(args.ablation_base, args.best_by)
    else:
        load_dir = args.load_dir
    
    # Validate model directory
    if not validate_model_dir(load_dir):
        sys.exit(1)
    
    # Validate data paths
    if not args.data_root.exists():
        print(f"Error: Data root not found: {args.data_root}")
        sys.exit(1)
    
    if not args.task_root.exists():
        print(f"Error: Task root not found: {args.task_root}")
        sys.exit(1)
    
    # All tasks
    tasks = ["emotion", "gambling", "language", "motor", "relational", "social", "WM"]
    
    # Create results directory
    results_dir = args.results_dir / "fm"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("Evaluating best model on all tasks")
    print("=" * 80)
    print(f"Model: {load_dir}")
    print(f"Data root: {args.data_root}")
    print(f"Task root: {args.task_root}")
    print(f"Results dir: {results_dir}")
    print(f"Tasks: {', '.join(tasks)}")
    print("=" * 80)
    print()
    
    # Run evaluation for each task
    success_count = 0
    fail_count = 0
    failed_tasks = []
    
    for task in tasks:
        save_dir = results_dir / f"best_model_{task}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[{task}] Evaluating...")
        print(f"  Save dir: {save_dir}")
        
        success = run_evaluation(
            load_dir=load_dir,
            task=task,
            data_root=args.data_root,
            task_root=args.task_root,
            save_dir=save_dir,
            use_evs=args.use_evs,
            ev_root=args.ev_root,
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
            failed_tasks.append(task)
        
        print()
    
    # Summary
    print("=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    print(f"Total tasks: {len(tasks)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    
    if failed_tasks:
        print(f"Failed tasks: {', '.join(failed_tasks)}")
    
    print()
    print(f"Results saved to: {results_dir}/best_model_<task>/")
    print()
    
    # Collect results
    if not args.skip_collect:
        print("Collecting results...")
        collect_script = Path(__file__).parent / "collect_fm_results.py"
        result = subprocess.run(
            [sys.executable, str(collect_script),
             "--search_dir", str(results_dir),
             "--output_dir", str(args.results_dir)],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("Results collected successfully!")
            print(f"\nView summary:")
            print(f"  cat {args.results_dir}/fm_results_summary.txt")
            print(f"  cat {args.results_dir}/fm_results.csv")
        else:
            print("Warning: Failed to collect results")
            print(result.stderr)
    
    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
