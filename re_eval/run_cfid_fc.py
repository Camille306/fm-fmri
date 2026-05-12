#!/usr/bin/env python3
"""
Conditional Fréchet distance using FC features (cFID-FC).

Uses the same pipeline as the discriminative score: load the trained fm-fmri
model and test set, run inference to get *paired* (real, generated) task
windows (same rest conditioning per pair). Then:

  f(x) = vec(FC(x))  with FC(x) = correlation matrix, vec = upper triangle
  cFID_FC = ||μ_r - μ_g||² + Tr(C_r + C_g - 2 (C_r C_g)^{1/2})

where μ_r, C_r and μ_g, C_g are mean and covariance of f(real) and f(generated)
over the paired samples. Lower cFID-FC = better.

Usage:
    python re_eval/run_cfid_fc.py \
        --load_dir /path/to/best/model \
        --data_root /path/to/hcp-resting-fc \
        --task_root /path/to/hcp-task-ts \
        --task_name emotion
"""

import os
import argparse
from pathlib import Path

import numpy as np
import torch

# Reuse loader and inference from discriminative score (same data, same pairing)
from run_discriminative_score import load_fmts_and_dataset, collect_real_and_generated

from fc_utils import cfid_fc


def main():
    p = argparse.ArgumentParser(description="cFID-FC: conditional Fréchet distance on FC features")
    # Data (default: HCP; override with env DATA_ROOT / TASK_ROOT or flags)
    p.add_argument("--data_root", type=str, default=os.getenv("DATA_ROOT", "./data/hcp-resting-fc"),
                   help="HCP resting data root (default: HCP path or DATA_ROOT)")
    p.add_argument("--task_root", type=str, default=os.getenv("TASK_ROOT", "./data/hcp-task-ts"),
                   help="HCP task data root (default: HCP path or TASK_ROOT)")
    p.add_argument("--task_name", type=str, default="emotion")
    p.add_argument("--use_evs", action="store_true")
    p.add_argument("--ev_root", type=str, default=None)
    p.add_argument("--lookback_length", type=int, default=512)
    p.add_argument("--prediction_length", type=int, default=None)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--batch_size", type=int, default=16)
    # Model
    p.add_argument("--load_dir", type=str, required=True, help="Path to directory containing best_fmts.pth")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--ode_steps", type=int, default=50)
    # FMTS architecture (overridden by checkpoint if present)
    p.add_argument("--rest_hidden", type=int, default=256)
    p.add_argument("--ctx_dim", type=int, default=256)
    p.add_argument("--t_dim", type=int, default=128)
    p.add_argument("--rest_encoder", type=str, default="transformer")
    p.add_argument("--rest_patch_len", type=int, default=16)
    p.add_argument("--rest_num_layers", type=int, default=2)
    p.add_argument("--rest_nhead", type=int, default=4)
    p.add_argument("--rest_dim_feedforward", type=int, default=512)
    p.add_argument("--prior_K", type=int, default=8)
    p.add_argument("--use_prior_detach", action="store_true")
    p.add_argument("--num_conditions", type=int, default=32)
    p.add_argument("--d_ev", type=int, default=64)
    p.add_argument("--use_hrf_kernel", action="store_true")
    p.add_argument("--hrf_kernel_len", type=int, default=20)
    p.add_argument("--hrf_num_basis", type=int, default=3)
    p.add_argument("--hrf_per_roi", action="store_true")
    p.add_argument("--use_ev_hrf_timecourse", action="store_true")
    p.add_argument("--ev_hrf_kernel_len", type=int, default=20)
    p.add_argument("--ev_hrf_num_basis", type=int, default=3)
    p.add_argument("--no_ev_hrf_delay_width", action="store_true")
    p.add_argument("--ev_hrf_smooth_boxcar", action="store_true")
    p.add_argument("--ev_hrf_boxcar_sigma", type=float, default=0.5)
    # cFID-FC
    p.add_argument("--eps", type=float, default=1e-6, help="Covariance regularization for numerical stability")
    p.add_argument("--max_fc_dim", type=int, default=500,
                   help="Project FC vectors to this many random dims before cFID (default: 500; set 0 to use full d=V(V-1)/2)")
    p.add_argument("--seed", type=int, default=42, help="Seed for random projection when using --max_fc_dim")
    p.add_argument("--save_dir", type=str, default=None, help="Optional: save cFID-FC value to file")

    args = p.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading FM model and test data...")
    model, test_loader, pred_len, V, args = load_fmts_and_dataset(args)
    print(f"Test windows: {len(test_loader.dataset)}, T={pred_len}, V={V}")

    print("Running inference to collect paired real and generated task series...")
    X_real, X_gen = collect_real_and_generated(
        model, test_loader, pred_len, args.device, ode_steps=args.ode_steps
    )
    n = X_real.shape[0]
    print(f"Collected {n} paired windows. Shape each: (N={n}, T={pred_len}, V={V})")

    rng = np.random.default_rng(getattr(args, "seed", 42))
    max_fc_dim = getattr(args, "max_fc_dim", 500)
    if max_fc_dim <= 0:
        max_fc_dim = None  # use full FC dimension
    print("Computing cFID-FC (FC feature distribution over paired samples)...")
    if max_fc_dim is not None:
        print(f"  Using random projection to {max_fc_dim} dims (default for large ROI count).")
    cfid = cfid_fc(X_real, X_gen, eps=args.eps, max_fc_dim=max_fc_dim, rng=rng)

    print()
    print("=" * 60)
    print("cFID-FC result")
    print("=" * 60)
    print(f"cFID-FC: {cfid:.6f}")
    print("  (lower = better; measures FC distribution match on conditioned pairs)")
    print("=" * 60)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        out_path = Path(args.save_dir) / "cfid_fc.txt"
        with open(out_path, "w") as f:
            f.write(f"cfid_fc\t{cfid:.6f}\n")
        print(f"Saved result to {out_path}")

    return cfid


if __name__ == "__main__":
    main()
