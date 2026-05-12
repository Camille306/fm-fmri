"""
Train STAGIN on Biopoint (1) real-only and (2) real + fm-fmri synthetic;
evaluate both and compare metrics to test whether synthetic data improves autism classification.

Usage:
  cd stagin_biopoint
  python run_real_vs_synthetic_comparison.py \\
    --data_root /path/to/biopoint_data \\
    --result_dir ./comparison_results \\
    --fm_checkpoint /path/to/results_flow_matching_biopoint/best_fmts.pth \\
    [--generate] [--ts_filename_suffix _shen268_ts.npy]

Steps:
  1. Optionally generate synthetic (if --generate or synthetic_dir empty)
  2. Train+test STAGIN on real-only -> result_dir/real_only
  3. Train+test STAGIN on real+synthetic -> result_dir/real_plus_synthetic
  4. Compare metric.csv and print/save comparison
"""

import os
import sys
import argparse
import subprocess
import pandas as pd
from pathlib import Path

# Run from stagin_biopoint so imports work
_stagin_dir = Path(__file__).resolve().parent
os.chdir(_stagin_dir)
sys.path.insert(0, str(_stagin_dir))


def parse():
    p = argparse.ArgumentParser(description="Compare STAGIN: real-only vs real+synthetic (fm-fmri)")
    p.add_argument("--data_root", type=str, default="./data/biopoint_data", help="Biopoint data root")
    p.add_argument("--csv_path", type=str, default=".//fMRI_autism/v3_causality_lstm/causal_lstm_2025/biopoint_data.csv", help="Biopoint subject CSV (subject_id, group)")
    p.add_argument("--result_dir", type=str, default="./comparison_results", help="Output: result_dir/real_only, result_dir/real_plus_synthetic, result_dir/comparison.csv")
    p.add_argument("--fm_checkpoint", type=str, required=True, help="Path to best_fmts.pth from run_flow_matching_biopoint")
    p.add_argument("--synthetic_dir", type=str, default=None, help="Where to save/load synthetic; default result_dir/synthetic_biopoint")
    p.add_argument("--ts_filename_suffix", type=str, default="_shen268_ts.npy", help="Must match fm-fmri training (e.g. _shen268_ts.npy or _aal_ts.npy)")
    p.add_argument("--generate", action="store_true", help="Generate synthetic from fm_checkpoint if missing")
    p.add_argument("--eprime_root", type=str, default=None, help="For synthetic generation with EV")
    p.add_argument("--k_fold", type=int, default=1, help="CV folds (1 = single train/test split)")
    p.add_argument("--train_ratio", type=float, default=0.8, help="Train fraction when k_fold=1")
    p.add_argument("--dynamic_length", type=int, default=None)
    p.add_argument("--num_epochs", type=int, default=40)
    p.add_argument("--minibatch_size", type=int, default=8)
    return p.parse_args()


def generate_synthetic(args):
    synthetic_dir = args.synthetic_dir or os.path.join(args.result_dir, "synthetic_biopoint")
    manifest = os.path.join(synthetic_dir, "synthetic_manifest.csv")
    if args.generate or not os.path.isfile(manifest):
        os.makedirs(synthetic_dir, exist_ok=True)
        cmd = [
            sys.executable,
            "generate_synthetic_biopoint.py",
            "--data_root", args.data_root,
            "--fm_checkpoint", args.fm_checkpoint,
            "--synthetic_dir", synthetic_dir,
            "--ts_filename_suffix", args.ts_filename_suffix,
        ]
        if args.csv_path:
            cmd += ["--csv_path", args.csv_path]
        if args.eprime_root:
            cmd += ["--eprime_root", args.eprime_root]
        print("Generating synthetic...", cmd)
        subprocess.run(cmd, check=True, cwd=_stagin_dir)
    return args.synthetic_dir or os.path.join(args.result_dir, "synthetic_biopoint")


def run_stagin(args, use_synthetic: bool, target_subdir: str):
    from util.option import parse as parse_opts
    from experiment import train, test

    # Build argv-like object
    class Argv:
        pass
    argv = Argv()
    argv.sourcedir = args.data_root
    argv.csv_path = args.csv_path
    argv.targetdir = os.path.join(args.result_dir, target_subdir)
    argv.synthetic_dir = args.synthetic_dir or os.path.join(args.result_dir, "synthetic_biopoint")
    argv.ts_filename_suffix = args.ts_filename_suffix
    argv.k_fold = args.k_fold
    argv.train_ratio = getattr(args, "train_ratio", 0.8)
    argv.dynamic_length = args.dynamic_length
    argv.num_epochs = args.num_epochs
    argv.minibatch_size = args.minibatch_size
    argv.use_synthetic = use_synthetic
    argv.seed = 0
    argv.exp_name = target_subdir
    argv.num_workers = 0
    argv.validate = True
    argv.hidden_dim = 128
    argv.num_heads = 1
    argv.num_layers = 4
    argv.sparsity = 30
    argv.dropout = 0.5
    argv.cls_token = "sum"
    argv.readout = "sero"
    argv.lr = 0.0005
    argv.max_lr = 0.001
    argv.reg_lambda = 0.00001
    argv.clip_grad = 0.0
    argv.window_size = 50
    argv.window_stride = 3

    os.makedirs(argv.targetdir, exist_ok=True)
    print(f"\n{'='*60}\nSTAGIN: {target_subdir}\n{'='*60}")
    train(argv)
    test(argv)
    return argv.targetdir


def load_metrics(targetdir):
    path = os.path.join(targetdir, "metric.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    # Last row is often std; we want mean metrics. Typically fold 0,1,..., then a row with std.
    # Take mean across folds (rows that are numeric fold indices)
    numeric = df[df["fold"].apply(lambda x: str(x).isdigit())]
    if len(numeric) == 0:
        return df.iloc[0].to_dict() if len(df) else None
    return numeric.mean(numeric_only=True).to_dict()


def main():
    args = parse()
    os.makedirs(args.result_dir, exist_ok=True)

    synthetic_dir = generate_synthetic(args)
    args.synthetic_dir = synthetic_dir

    run_stagin(args, use_synthetic=False, target_subdir="real_only")
    run_stagin(args, use_synthetic=True, target_subdir="real_plus_synthetic")

    m_real = load_metrics(os.path.join(args.result_dir, "real_only"))
    m_syn = load_metrics(os.path.join(args.result_dir, "real_plus_synthetic"))

    print("\n" + "=" * 60)
    print("COMPARISON: Real-only vs Real + Synthetic")
    print("=" * 60)

    if m_real is None or m_syn is None:
        print("Could not load metric.csv from one or both runs.")
        return

    comparison = []
    for key in sorted(m_real.keys()):
        if key in m_syn and isinstance(m_real[key], (int, float)) and isinstance(m_syn[key], (int, float)):
            delta = m_syn[key] - m_real[key]
            comparison.append({"metric": key, "real_only": m_real[key], "real_plus_synthetic": m_syn[key], "delta": delta})
            print(f"  {key}: real_only={m_real[key]:.4f}  real_plus_synthetic={m_syn[key]:.4f}  delta={delta:+.4f}")

    out_path = os.path.join(args.result_dir, "comparison.csv")
    pd.DataFrame(comparison).to_csv(out_path, index=False)
    print(f"\nComparison saved to {out_path}")


if __name__ == "__main__":
    main()
