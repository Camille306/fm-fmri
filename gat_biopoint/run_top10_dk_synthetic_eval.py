#!/usr/bin/env python3
"""
Run GAT Biopoint synthetic-data evaluation for TOP-K FM DK runs (no SLURM).

Pipeline:
  0) Train/test GAT with real-only data as baseline (run once)
  1) Per FM run: generate synthetic task series from FM checkpoint
  2) Per FM run: train/test GAT with real+synthetic data
  3) Parse GAT test_results.txt and collect ACC/F1/AUC
  4) Write comparison summary (baseline vs each synthetic augmentation)

Usage (from repo root):
  python gat_biopoint/run_top10_dk_synthetic_eval.py \
    --fm_results_dir ./results_fm_biopoint_dk_tuning_v2 \
    --top_k 10
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


def _read_topk_rows(summary_csv: Path, top_k: int) -> list[dict]:
    with open(summary_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if "rank" in r and "dir" in r]
    rows.sort(key=lambda r: int(r["rank"]))
    return rows[:top_k]


def _parse_gat_test_results(path: Path) -> dict[str, float]:
    out = {"acc": float("nan"), "f1": float("nan"), "auc": float("nan")}
    if not path.is_file():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            key, val = parts[0].strip().lower(), parts[1].strip()
            if key in out:
                try:
                    out[key] = float(val)
                except ValueError:
                    pass
    return out


def _run(cmd: list[str], cwd: Path) -> None:
    print("  $ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _supports_cli_flags(python_exe: str, script_relpath: str, flags: list[str], cwd: Path) -> bool:
    """Return True iff all flags are present in `<script> -h` output."""
    proc = subprocess.run(
        [python_exe, script_relpath, "-h"],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    help_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return all(flag in help_text for flag in flags)


def _gat_common_args(args) -> list[str]:
    """GAT CLI arguments shared by both real-only and synthetic runs."""
    return [
        "--device", args.device,
        "--sourcedir", args.data_root,
        "--csv_path", args.csv_path,
        "--atlas_source", "dk",
        "--dk_atlas_ts_root", args.dk_atlas_ts_root,
        "--k_fold", str(args.gat_k_fold),
        "--num_epochs", str(args.gat_num_epochs),
        "--patience", str(args.gat_patience),
        "--lr", str(args.gat_lr),
        "--hidden_dim", str(args.gat_hidden_dim),
        "--dropout", str(args.gat_dropout),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate top-K FM DK synthetic datasets with GAT (real+synthetic vs real-only baseline)"
    )
    parser.add_argument(
        "--fm_results_dir",
        type=str,
        default=".//results_fm_biopoint_dk_tuning_v2",
        help="FM DK tuning results directory containing subdirs + param_search_summary.csv",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default=None,
        help="Optional explicit path to param_search_summary.csv (default: <fm_results_dir>/param_search_summary.csv)",
    )
    parser.add_argument("--top_k", type=int, default=10, help="Number of top FM runs to evaluate")

    parser.add_argument(
        "--data_root",
        type=str,
        default="./data/biopoint_data",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="./data/biopoint_data.csv",
    )
    parser.add_argument(
        "--eprime_root",
        type=str,
        default="~/project_pi/user/rest_to_task/biopoint/eprime_biopoint",
    )
    parser.add_argument(
        "--dk_atlas_ts_root",
        type=str,
        default="./data/biopoint_dk_atlas",
    )
    parser.add_argument("--ode_steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])

    parser.add_argument(
        "--synthetic_base",
        type=str,
        default=None,
        help="Where to write synthetic datasets (default: <repo>/gat_biopoint/synthetic_topk_dk)",
    )
    parser.add_argument(
        "--gat_results_base",
        type=str,
        default=None,
        help="Where to write GAT outputs (default: <repo>/gat_biopoint/results_topk_dk_syn)",
    )
    parser.add_argument(
        "--summary_out",
        type=str,
        default=None,
        help="Where to write final ranking CSV (default: <gat_results_base>/topk_synthetic_eval_summary.csv)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip a rank if its GAT test_results.txt already exists",
    )

    # GAT training hyperparameters
    parser.add_argument("--gat_k_fold", type=int, default=5, help="GAT k-fold CV")
    parser.add_argument("--gat_num_epochs", type=int, default=50, help="GAT max epochs per fold")
    parser.add_argument("--gat_patience", type=int, default=10, help="GAT early stopping patience")
    parser.add_argument("--gat_lr", type=float, default=1e-4, help="GAT learning rate")
    parser.add_argument("--gat_hidden_dim", type=int, default=128, help="GAT hidden dimension")
    parser.add_argument("--gat_dropout", type=float, default=0.2, help="GAT dropout rate")

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    fm_results_dir = Path(os.path.expanduser(args.fm_results_dir)).resolve()
    summary_csv = Path(os.path.expanduser(args.summary_csv)).resolve() if args.summary_csv else (fm_results_dir / "param_search_summary.csv")

    if not summary_csv.is_file():
        raise FileNotFoundError(
            f"Summary CSV not found: {summary_csv}\n"
            f"Run: python biopoint/slurm/collect_fm_biopoint_param_search.py --results_dir {fm_results_dir} --top_n {args.top_k}"
        )

    synthetic_base = Path(os.path.expanduser(args.synthetic_base)).resolve() if args.synthetic_base else (repo_root / "gat_biopoint" / "synthetic_topk_dk")
    gat_results_base = Path(os.path.expanduser(args.gat_results_base)).resolve() if args.gat_results_base else (repo_root / "gat_biopoint" / "results_topk_dk_syn")
    synthetic_base.mkdir(parents=True, exist_ok=True)
    gat_results_base.mkdir(parents=True, exist_ok=True)

    eprime_root = str(Path(os.path.expanduser(args.eprime_root)))

    if not _supports_cli_flags(
        python_exe=sys.executable,
        script_relpath="stagin_biopoint/generate_synthetic_biopoint.py",
        flags=["--atlas_source", "--dk_atlas_ts_root"],
        cwd=repo_root,
    ):
        raise RuntimeError(
            "Your stagin_biopoint/generate_synthetic_biopoint.py is an older version "
            "that does not support --atlas_source/--dk_atlas_ts_root.\n"
            "Please sync/update that file from the latest repo changes, then rerun."
        )

    # ── Step 0: real-only baseline (run once) ──
    baseline_exp_name = "gat_bp_dk_real_only"
    baseline_exp_dir = gat_results_base / baseline_exp_name
    baseline_results = baseline_exp_dir / "test_results.txt"

    print("\n" + "=" * 80)
    print("BASELINE: Real-only GAT training")
    print("=" * 80)

    if args.skip_existing and baseline_results.is_file():
        print(f"Skipping existing baseline: {baseline_results}")
    else:
        _run(
            [
                sys.executable,
                "gat_biopoint/main.py",
                "--train",
                "--test",
                "--exp_name",
                baseline_exp_name,
                "--targetdir",
                str(gat_results_base),
            ] + _gat_common_args(args),
            cwd=repo_root,
        )

    baseline_metrics = _parse_gat_test_results(baseline_results)
    print(
        f"Baseline (real-only): "
        f"acc={baseline_metrics['acc']:.4f}  f1={baseline_metrics['f1']:.4f}  auc={baseline_metrics['auc']:.4f}"
    )

    # ── Steps 1-3: per-FM-rank synthetic evaluation ──
    top_rows = _read_topk_rows(summary_csv, args.top_k)
    if not top_rows:
        raise RuntimeError(f"No rows found in {summary_csv}")

    collected: list[dict] = []

    for row in top_rows:
        rank = int(row["rank"])
        fm_dir_name = row["dir"]
        fm_ckpt = fm_results_dir / fm_dir_name / "best_fmts.pth"
        if not fm_ckpt.is_file():
            print(f"[skip] rank={rank}: missing checkpoint {fm_ckpt}")
            continue

        syn_dir = synthetic_base / f"rank{rank}_{fm_dir_name}"
        syn_dir.mkdir(parents=True, exist_ok=True)

        exp_name = f"gat_bp_dk_syn_rank{rank}"
        exp_dir = gat_results_base / exp_name
        exp_results = exp_dir / "test_results.txt"

        print("\n" + "=" * 80)
        print(f"Rank {rank}  FM run: {fm_dir_name}")
        print("=" * 80)

        if args.skip_existing and exp_results.is_file():
            print(f"Skipping existing experiment: {exp_results}")
        else:
            _run(
                [
                    sys.executable,
                    "stagin_biopoint/generate_synthetic_biopoint.py",
                    "--data_root",
                    args.data_root,
                    "--csv_path",
                    args.csv_path,
                    "--atlas_source",
                    "dk",
                    "--dk_atlas_ts_root",
                    args.dk_atlas_ts_root,
                    "--fm_checkpoint",
                    str(fm_ckpt),
                    "--synthetic_dir",
                    str(syn_dir),
                    "--eprime_root",
                    eprime_root,
                    "--ode_steps",
                    str(args.ode_steps),
                    "--device",
                    args.device,
                ],
                cwd=repo_root,
            )

            _run(
                [
                    sys.executable,
                    "gat_biopoint/main.py",
                    "--train",
                    "--test",
                    "--exp_name",
                    exp_name,
                    "--targetdir",
                    str(gat_results_base),
                    "--use_synthetic",
                    "--synthetic_dir",
                    str(syn_dir),
                ] + _gat_common_args(args),
                cwd=repo_root,
            )

        metrics = _parse_gat_test_results(exp_results)
        delta_acc = metrics["acc"] - baseline_metrics["acc"]
        delta_auc = metrics["auc"] - baseline_metrics["auc"]
        result_row = {
            "rank": rank,
            "fm_run_dir": fm_dir_name,
            "fm_ckpt": str(fm_ckpt),
            "synthetic_dir": str(syn_dir),
            "gat_exp_dir": str(exp_dir),
            "acc": metrics["acc"],
            "f1": metrics["f1"],
            "auc": metrics["auc"],
            "delta_acc_vs_baseline": delta_acc,
            "delta_auc_vs_baseline": delta_auc,
        }
        collected.append(result_row)
        print(
            f"Result rank={rank}: "
            f"acc={metrics['acc']:.4f}  f1={metrics['f1']:.4f}  auc={metrics['auc']:.4f}  "
            f"(vs baseline: acc {delta_acc:+.4f}  auc {delta_auc:+.4f})"
        )

    if not collected:
        raise RuntimeError("No results collected.")

    collected_sorted = sorted(
        collected,
        key=lambda r: (r["auc"], r["f1"], r["acc"]),
        reverse=True,
    )
    for i, r in enumerate(collected_sorted, start=1):
        r["gat_rank"] = i

    summary_out = Path(os.path.expanduser(args.summary_out)).resolve() if args.summary_out else (gat_results_base / "topk_synthetic_eval_summary.csv")
    fieldnames = [
        "gat_rank",
        "rank",
        "fm_run_dir",
        "acc",
        "f1",
        "auc",
        "delta_acc_vs_baseline",
        "delta_auc_vs_baseline",
        "fm_ckpt",
        "synthetic_dir",
        "gat_exp_dir",
    ]
    with open(summary_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Write baseline row first
        writer.writerow({
            "gat_rank": 0,
            "rank": "baseline",
            "fm_run_dir": "real_only",
            "acc": baseline_metrics["acc"],
            "f1": baseline_metrics["f1"],
            "auc": baseline_metrics["auc"],
            "delta_acc_vs_baseline": 0.0,
            "delta_auc_vs_baseline": 0.0,
            "fm_ckpt": "",
            "synthetic_dir": "",
            "gat_exp_dir": str(baseline_exp_dir),
        })
        writer.writerows(collected_sorted)

    print("\n" + "=" * 80)
    print("Top-K synthetic evaluation complete.")
    print(f"Summary CSV: {summary_out}")
    print(
        f"\nBaseline (real-only): "
        f"ACC={baseline_metrics['acc']:.4f}, F1={baseline_metrics['f1']:.4f}, AUC={baseline_metrics['auc']:.4f}"
    )
    best = collected_sorted[0]
    print(
        f"Best synthetic:       FM rank={best['rank']} ({best['fm_run_dir']}) "
        f"-> ACC={best['acc']:.4f}, F1={best['f1']:.4f}, AUC={best['auc']:.4f}"
    )
    print(
        f"Improvement:          ACC {best['delta_acc_vs_baseline']:+.4f}, "
        f"AUC {best['delta_auc_vs_baseline']:+.4f}"
    )


if __name__ == "__main__":
    main()
