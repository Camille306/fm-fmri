#!/usr/bin/env python3
"""
Train and evaluate the Flow Matching (FM-TS) model for rest-to-task on Biopoint data
with paired eprime EV files (BioPoints_Timing=1, RandPoints_Timing=2).

Run from repo root:
  python biopoint/run_flow_matching_biopoint.py --save_dir ./results_flow_matching_biopoint
  # Default eprime: ~/project_pi/user/rest_to_task/biopoint/eprime_biopoint
  # (if missing, falls back to biopoint/eprime_biopoint or biopoint/eprime_timing_download in the repo).
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_biopoint_dir = Path(__file__).resolve().parent
_repo_root = _biopoint_dir.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_biopoint_dir))

# Default eprime tree on cluster (user home); expanded so ~ works.
_DEFAULT_EPRIME_ROOT = os.path.expanduser(
    "~/project_pi/user/rest_to_task/biopoint/eprime_biopoint"
)

from biopoint_dataset import BiopointDatasetAdapter
from biopoint_window_dataset import BiopointWindowDataset


def get_args():
    p = argparse.ArgumentParser(description="Train Flow Matching (FM-TS) on Biopoint rest-to-task with EV")
    p.add_argument("--data_root", type=str, default="./data/biopoint_data")
    p.add_argument("--csv_path", type=str, default="./data/biopoint_data.csv")
    p.add_argument(
        "--eprime_root",
        type=str,
        default=_DEFAULT_EPRIME_ROOT,
        help=(
            "Root for eprime EV files: eprime_root/{subject_id}/.../BioPoints_Timing and RandPoints_Timing. "
            f"Default: {_DEFAULT_EPRIME_ROOT} (falls back to biopoint/eprime_* in repo if missing)."
        ),
    )
    p.add_argument("--atlas_source", type=str, default="dk", choices=["dk", "shen268"],
                   help="Which atlas ROI time-series to use (default: dk)")
    p.add_argument("--dk_atlas_ts_root", type=str, default="./data/biopoint_dk_atlas",
                   help="Root containing <subject_id>_rest_roi_ts.pt and <subject_id>_task_roi_ts.pt (when atlas_source=dk)")
    p.add_argument("--save_dir", type=str, default="./results_flow_matching_biopoint")
    p.add_argument("--lookback_length", type=int, default=200)
    p.add_argument("--prediction_length", type=int, default=None)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--normalize", action="store_true", default=True)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--eval_only", action="store_true", help="Load best checkpoint and run test eval + viz only")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    # FM-TS model
    p.add_argument("--rest_hidden", type=int, default=256)
    p.add_argument("--ctx_dim", type=int, default=256)
    p.add_argument("--t_dim", type=int, default=128)
    p.add_argument("--num_conditions", type=int, default=32)
    p.add_argument("--d_ev", type=int, default=64)
    p.add_argument("--rest_nhead", type=int, default=4, help="Number of attention heads in rest transformer encoder")
    p.add_argument("--rest_num_layers", type=int, default=2, help="Number of layers in rest transformer encoder")
    p.add_argument("--prior_K", type=int, default=8, help="Low-rank dimension for cross-ROI prior coupling")
    p.add_argument("--ode_steps", type=int, default=50)
    # Auxiliary losses (Frequency + FC + Coherence)
    p.add_argument(
        "--freq_loss_weight",
        type=float,
        default=0.1,
        help="Weight for PSD/frequency-domain auxiliary loss (0.01–0.05 Hz band)",
    )
    p.add_argument(
        "--fc_loss_weight",
        type=float,
        default=0.1,
        help="Weight for functional-connectivity (correlation matrix) loss",
    )
    p.add_argument(
        "--coh_loss_weight",
        type=float,
        default=0.0,
        help="Weight for coherence loss (frequency-specific inter-ROI coupling)",
    )
    p.add_argument("--aux_ode_steps", type=int, default=10)
    p.add_argument("--fc_strength_power", type=float, default=2.0,
                   help="Exponent for strength-weighting in FC loss (higher = more weight on strong edges)")
    return p.parse_args()


def build_dataloaders(args):
    adapter = BiopointDatasetAdapter(
        data_root=args.data_root,
        csv_path=args.csv_path,
        ev_root=args.eprime_root,
        atlas_source=args.atlas_source,
        dk_atlas_ts_root=args.dk_atlas_ts_root,
    )
    n_subj = len(adapter.subject_ids)
    print(f"Biopoint adapter (with EV): {n_subj} subjects")
    print(f"  EV root: {args.eprime_root}")

    if n_subj == 0:
        raise ValueError(
            "No subjects left after BiopointDatasetAdapter filtering (need DK/Shen ROI time series "
            "+ matching eprime EV files). See diagnostic lines above; check --dk_atlas_ts_root, "
            "--data_root/--csv_path subject IDs, and --eprime_root layout."
        )

    min_rest_len = min(adapter.load_subject(sid).shape[0] for sid in adapter.subject_ids)
    min_task_len = min(adapter.load_task_subject(sid).shape[0] for sid in adapter.subject_ids)

    if args.prediction_length is None:
        args.prediction_length = min_task_len
        print(f"Inferred prediction_length={args.prediction_length}")
    args.prediction_length = min(args.prediction_length, min_task_len)

    lookback_length = min(args.lookback_length, min_rest_len)
    if lookback_length < 1:
        lookback_length = 1
    if args.prediction_length < 1:
        args.prediction_length = 1
    args.lookback_length = lookback_length

    print(f"Using lookback_length={args.lookback_length}, prediction_length={args.prediction_length} (data: min_rest={min_rest_len}, min_task={min_task_len})")

    train_ds = BiopointWindowDataset(
        adapter,
        lookback_length=args.lookback_length,
        prediction_length=args.prediction_length,
        stride=args.stride,
        normalize=args.normalize,
        split="train",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        use_evs=True,
    )
    val_ds = BiopointWindowDataset(
        adapter,
        lookback_length=args.lookback_length,
        prediction_length=args.prediction_length,
        stride=args.stride,
        normalize=args.normalize,
        split="val",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        use_evs=True,
    )
    test_ds = BiopointWindowDataset(
        adapter,
        lookback_length=args.lookback_length,
        prediction_length=args.prediction_length,
        stride=args.stride,
        normalize=args.normalize,
        split="test",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        use_evs=True,
    )

    if args.normalize and train_ds.rest_means is not None:
        val_ds.rest_means, val_ds.rest_stds = train_ds.rest_means, train_ds.rest_stds
        val_ds.task_means, val_ds.task_stds = train_ds.task_means, train_ds.task_stds
        test_ds.rest_means, test_ds.rest_stds = train_ds.rest_means, train_ds.rest_stds
        test_ds.task_means, test_ds.task_stds = train_ds.task_means, train_ds.task_stds

    if len(train_ds) == 0:
        raise ValueError(
            f"No train windows: lookback_length={args.lookback_length} and prediction_length={args.prediction_length} "
            f"may exceed your data (min_rest={min_rest_len}, min_task={min_task_len})."
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    sample = next(iter(train_loader))
    V = int(sample["input"].shape[-1])
    print(f"Windows: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}  V={V}")
    return train_loader, val_loader, test_loader, V


def _collect_paired_windows(model, test_loader, device, pred_len, ode_steps):
    """
    Run model inference over the test set once and return paired numpy arrays
    together with the subject ID for every window.

    Returns (X_real, X_gen, subject_ids):
      X_real, X_gen : (N, T, V) float64
      subject_ids   : list[str] of length N  (window → subject mapping)
    Returns (None, None, None) if the test set is empty.
    """
    model.eval()
    real_list, gen_list, sid_list = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x_rest = batch["input"].to(device)   # (B, L, V)
            x_task = batch["target"].to(device)  # (B, T, V)
            ev = batch.get("ev", None)
            ev_mask = batch.get("ev_mask", None)
            if ev is not None:
                ev = ev.to(device)
            if ev_mask is not None:
                ev_mask = ev_mask.to(device)

            B, T, V = x_task.shape
            try:
                x_gen = model.sample(x_rest, pred_len, steps=ode_steps, ev=ev, ev_mask=ev_mask)
            except TypeError:
                x_gen = model.sample(x_rest, pred_len, steps=ode_steps)

            T_use = min(T, pred_len, x_gen.shape[1])
            real_list.append(x_task[:, :T_use, :].cpu().numpy())
            gen_list.append(x_gen[:, :T_use, :].cpu().numpy())

            # subject_id may be a list of strings or a batch tensor of strings
            batch_sids = batch.get("subject_id", None)
            if batch_sids is None:
                batch_sids = ["unknown"] * B
            elif not isinstance(batch_sids, (list, tuple)):
                # tensor of bytes / object — convert to list of str
                batch_sids = [str(s) for s in batch_sids]
            sid_list.extend(list(batch_sids))

    if not real_list:
        return None, None, None

    X_real = np.concatenate(real_list, axis=0).astype(np.float64)  # (N, T, V)
    X_gen  = np.concatenate(gen_list,  axis=0).astype(np.float64)  # (N, T, V)
    return X_real, X_gen, sid_list


def compute_fc_metrics(model, test_loader, device, pred_len, ode_steps, max_fc_dim=500):
    """
    Run inference once and compute:
      - cFID-FC          : population-level Fréchet distance (all windows pooled)
      - cFID-FC (subj)   : per-subject Fréchet distance, mean ± std across subjects
      - FC-Precision@5%  : mean fraction of top-5% FC edges recovered per window

    Returns dict with keys:
        'cfid'            : float  population cFID (lower = better)
        'cfid_subj_mean'  : float  mean of per-subject cFIDs
        'cfid_subj_std'   : float  std  of per-subject cFIDs
        'cfid_subj_n'     : int    number of subjects scored
        'cfid_subj_skip'  : int    subjects skipped (too few windows)
        'fc_prec5'        : float  FC-Precision@5% (higher = better, max=1.0)
    """
    _re_eval_dir = _repo_root / "re_eval"
    sys.path.insert(0, str(_re_eval_dir))
    try:
        import fc_utils
        cfid_fc = fc_utils.cfid_fc
        compute_fc_precision_at_5_paired = fc_utils.compute_fc_precision_at_5_paired
        cfid_fc_subject_level = getattr(fc_utils, "cfid_fc_subject_level", None)
    except Exception as e:
        print(f"  [FC metrics] Could not import required fc_utils functions: {e}. Skipping.")
        nan = float("nan")
        return {"cfid": nan, "cfid_subj_mean": nan, "cfid_subj_std": nan,
                "cfid_subj_n": 0, "cfid_subj_skip": 0, "fc_prec5": nan}

    print("  Collecting paired windows for FC metrics (single inference pass)...")
    X_real, X_gen, subject_ids = _collect_paired_windows(
        model, test_loader, device, pred_len, ode_steps
    )

    nan = float("nan")
    if X_real is None:
        print("  [FC metrics] No test windows collected.")
        return {"cfid": nan, "cfid_subj_mean": nan, "cfid_subj_std": nan,
                "cfid_subj_n": 0, "cfid_subj_skip": 0, "fc_prec5": nan}

    if X_real.shape[0] < 2:
        print("  [FC metrics] Too few test windows (need ≥ 2). Skipping.")
        return {"cfid": nan, "cfid_subj_mean": nan, "cfid_subj_std": nan,
                "cfid_subj_n": 0, "cfid_subj_skip": 0, "fc_prec5": nan}

    rng = np.random.default_rng(0)

    # Population-level cFID (all windows pooled)
    try:
        cfid_score = cfid_fc(X_real, X_gen, max_fc_dim=max_fc_dim, rng=rng)
    except Exception as ex:
        print(f"  [cFID-pop] Computation failed: {ex}")
        cfid_score = nan

    # Subject-level cFID (per-subject Fréchet, then mean ± std)
    cfid_subj_mean = nan
    cfid_subj_std  = nan
    cfid_subj_n    = 0
    cfid_subj_skip = 0
    if cfid_fc_subject_level is None:
        print("  [cFID-subj] fc_utils.cfid_fc_subject_level not available; skipping subject-level cFID.")
    else:
        try:
            subj_result = cfid_fc_subject_level(
                X_real, X_gen, subject_ids,
                max_fc_dim=max_fc_dim,
                rng=np.random.default_rng(0),  # fresh rng with same seed for same projection
            )
            cfid_subj_mean = subj_result["mean"]
            cfid_subj_std  = subj_result["std"]
            cfid_subj_n    = subj_result["n_subjects"]
            cfid_subj_skip = subj_result["n_skipped"]
            print(
                f"  [cFID-subj] {cfid_subj_n} subjects scored, "
                f"{cfid_subj_skip} skipped (too few windows): "
                f"mean={cfid_subj_mean:.4f}  std={cfid_subj_std:.4f}"
            )
        except Exception as ex:
            print(f"  [cFID-subj] Computation failed: {ex}")

    # FC-Precision@5%
    try:
        prec5 = compute_fc_precision_at_5_paired(X_real, X_gen)
    except Exception as ex:
        print(f"  [FC-Prec@5%] Computation failed: {ex}")
        prec5 = nan

    return {
        "cfid":           cfid_score,
        "cfid_subj_mean": cfid_subj_mean,
        "cfid_subj_std":  cfid_subj_std,
        "cfid_subj_n":    cfid_subj_n,
        "cfid_subj_skip": cfid_subj_skip,
        "fc_prec5":       prec5,
    }


def compute_cfid(model, test_loader, device, pred_len, ode_steps, max_fc_dim=500):
    """Backwards-compatible wrapper — returns only the cFID scalar."""
    return compute_fc_metrics(model, test_loader, device, pred_len, ode_steps,
                              max_fc_dim=max_fc_dim)["cfid"]


def main():
    args = get_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Empty string (e.g. from shell) → use same default as argparse
    er = (args.eprime_root or "").strip()
    if not er:
        er = _DEFAULT_EPRIME_ROOT
    args.eprime_root = os.path.abspath(os.path.expanduser(er))

    if not os.path.isdir(args.eprime_root):
        for name in ("eprime_biopoint", "eprime_timing_download"):
            candidate = os.path.join(_biopoint_dir, name)
            if os.path.isdir(candidate):
                fallback = os.path.abspath(candidate)
                print(
                    f"eprime_root not found: {args.eprime_root!r}; using {fallback!r}"
                )
                args.eprime_root = fallback
                break
        else:
            raise ValueError(
                f"eprime_root is not a directory: {args.eprime_root!r}. "
                "Set --eprime_root or add biopoint/eprime_biopoint or biopoint/eprime_timing_download."
            )
    else:
        print(f"Using eprime_root: {args.eprime_root}")

    print(f"Flow Matching (FM-TS)  device: {args.device}  save_dir: {args.save_dir}")

    # If evaluating, prefer atlas settings stored in the checkpoint args.
    best_path = os.path.join(args.save_dir, "best_fmts.pth")
    if args.eval_only and os.path.isfile(best_path):
        try:
            ckpt = torch.load(best_path, map_location="cpu")
            saved_args = ckpt.get("args") or {}
            args.atlas_source = saved_args.get("atlas_source", args.atlas_source)
            args.dk_atlas_ts_root = saved_args.get("dk_atlas_ts_root", args.dk_atlas_ts_root)
            print(f"Loaded atlas settings from checkpoint: atlas_source={args.atlas_source!r}")
        except Exception:
            pass

    train_loader, val_loader, test_loader, V = build_dataloaders(args)

    _fm_fmri_dir = _repo_root / "fm-fmri"
    if not _fm_fmri_dir.exists():
        raise ImportError(f"fm-fmri not found at {_fm_fmri_dir}; flow matching requires fm-fmri.")
    sys.path.insert(0, str(_fm_fmri_dir))
    from fm_fmri import (
        FMTS,
        train_epoch,
        evaluate_subject_level_dedup,
        evaluate_subject_level_dedup_with_best_subject,
    )
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR

    os.makedirs(args.save_dir, exist_ok=True)

    model = FMTS(
        v_dim=V,
        rest_hidden=args.rest_hidden,
        ctx_dim=args.ctx_dim,
        t_dim=args.t_dim,
        use_evs=True,
        num_conditions=args.num_conditions,
        d_ev=args.d_ev,
        rest_nhead=args.rest_nhead,
        rest_num_layers=args.rest_num_layers,
        prior_K=args.prior_K,
    ).to(args.device)
    print(f"FM-TS use_evs=True  params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    opt = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    best_path = os.path.join(args.save_dir, "best_fmts.pth")

    if args.eval_only:
        ckpt = torch.load(best_path, map_location=args.device)
        model.load_state_dict(ckpt["model"])
        test_metrics = evaluate_subject_level_dedup_with_best_subject(
            model, test_loader, args.device,
            pred_len=args.prediction_length,
            ode_steps=args.ode_steps,
            out_dir=args.save_dir,
            fs=0.72,
        )
        print("  Computing cFID-FC and FC-Precision@5% on test set...")
        fc_scores = compute_fc_metrics(model, test_loader, args.device,
                                       pred_len=args.prediction_length,
                                       ode_steps=args.ode_steps)
        cfid_score     = fc_scores["cfid"]
        cfid_s_mean    = fc_scores["cfid_subj_mean"]
        cfid_s_std     = fc_scores["cfid_subj_std"]
        cfid_s_n       = fc_scores["cfid_subj_n"]
        cfid_s_skip    = fc_scores["cfid_subj_skip"]
        prec5          = fc_scores["fc_prec5"]
        print("\n" + "=" * 60)
        print("Flow matching (FM-TS) TEST (Biopoint, subject-level dedup)")
        print("=" * 60)
        print(f"MSE:                 {test_metrics['mse']:.6f} ± {test_metrics['mse_std']:.6f}")
        print(f"MAE:                 {test_metrics['mae']:.6f} ± {test_metrics['mae_std']:.6f}")
        print(f"PSD:                 {test_metrics['freq_diff']:.6f} ± {test_metrics['freq_diff_std']:.6f}")
        print(f"FC sim:              {test_metrics['fc_similarity']:.6f} ± {test_metrics['fc_similarity_std']:.6f}")
        print(f"cFID-FC (pop):       {cfid_score:.4f}  (lower = better, all windows pooled)")
        print(f"cFID-FC (subj mean): {cfid_s_mean:.4f} ± {cfid_s_std:.4f}  "
              f"(n={cfid_s_n} subjects, {cfid_s_skip} skipped)")
        print(f"FC-Prec@5%%:         {prec5:.4f}  (higher = better, max=1.0)")
        print(f"Num subjects:        {test_metrics['num_subjects']}")
        print("=" * 60)
        path = os.path.join(args.save_dir, "test_results.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("Flow matching (FM-TS) TEST (Biopoint, subject-level dedup)\n")
            f.write("=" * 60 + "\n")
            f.write(f"MSE (mean ± std): {test_metrics['mse']:.6f} ± {test_metrics['mse_std']:.6f}\n")
            f.write(f"MAE (mean ± std): {test_metrics['mae']:.6f} ± {test_metrics['mae_std']:.6f}\n")
            f.write(f"PSD (mean ± std): {test_metrics['freq_diff']:.6f} ± {test_metrics['freq_diff_std']:.6f}\n")
            f.write(f"FC sim (mean ± std): {test_metrics['fc_similarity']:.6f} ± {test_metrics['fc_similarity_std']:.6f}\n")
            f.write(f"cFID-FC (population): {cfid_score:.6f}\n")
            f.write(f"cFID-FC (subject mean): {cfid_s_mean:.6f}\n")
            f.write(f"cFID-FC (subject std): {cfid_s_std:.6f}\n")
            f.write(f"cFID-FC (subject n): {cfid_s_n}\n")
            f.write(f"cFID-FC (subject skipped): {cfid_s_skip}\n")
            f.write(f"FC-Precision@5%: {prec5:.6f}\n")
            f.write(f"Num subjects: {test_metrics['num_subjects']}\n")
        print(f"Test results written to {path}")
        return

    best_val_fc_sim = -float("inf")
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_fm, tr_freq, tr_fc, tr_coh = train_epoch(
            model, train_loader, opt, args.device,
            max_grad_norm=args.max_grad_norm,
            freq_loss_weight=args.freq_loss_weight,
            fc_loss_weight=args.fc_loss_weight,
            coh_loss_weight=args.coh_loss_weight,
            aux_ode_steps=args.aux_ode_steps,
            fc_strength_power=args.fc_strength_power,
        )
        val_metrics = evaluate_subject_level_dedup(
            model, val_loader, args.device,
            pred_len=args.prediction_length,
            ode_steps=args.ode_steps,
        )
        sched.step()
        val_fc = val_metrics.get('fc_similarity', 0.0)
        print(f"Epoch {ep}/{args.epochs}  train_loss={tr_loss:.6f}  val_mse={val_metrics['mse']:.6f}  val_fc_sim={val_fc:.6f}")
        if val_fc > best_val_fc_sim:
            best_val_fc_sim = val_fc
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)
            print(f"  Saved best (val_fc_sim={best_val_fc_sim:.6f}) -> {best_path}")

    ckpt = torch.load(best_path, map_location=args.device)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate_subject_level_dedup_with_best_subject(
        model, test_loader, args.device,
        pred_len=args.prediction_length,
        ode_steps=args.ode_steps,
        out_dir=args.save_dir,
        fs=0.72,
    )
    print("  Computing cFID-FC and FC-Precision@5% on test set...")
    fc_scores = compute_fc_metrics(model, test_loader, args.device,
                                   pred_len=args.prediction_length,
                                   ode_steps=args.ode_steps)
    cfid_score = fc_scores["cfid"]
    prec5      = fc_scores["fc_prec5"]
    print("\n" + "=" * 60)
    print("Flow matching (FM-TS) TEST (Biopoint, subject-level dedup)")
    print("=" * 60)
    print(f"MSE:          {test_metrics['mse']:.6f} ± {test_metrics['mse_std']:.6f}")
    print(f"MAE:          {test_metrics['mae']:.6f} ± {test_metrics['mae_std']:.6f}")
    print(f"PSD:          {test_metrics['freq_diff']:.6f} ± {test_metrics['freq_diff_std']:.6f}")
    print(f"FC sim:       {test_metrics['fc_similarity']:.6f} ± {test_metrics['fc_similarity_std']:.6f}")
    print(f"cFID-FC:      {cfid_score:.4f}  (lower = better)")
    print(f"FC-Prec@5%%:  {prec5:.4f}  (higher = better, max=1.0)")
    print(f"Num subjects: {test_metrics['num_subjects']}")
    print("=" * 60)

    path = os.path.join(args.save_dir, "test_results.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("Flow matching (FM-TS) TEST (Biopoint, subject-level dedup)\n")
        f.write("=" * 60 + "\n")
        f.write(f"MSE (mean ± std): {test_metrics['mse']:.6f} ± {test_metrics['mse_std']:.6f}\n")
        f.write(f"MAE (mean ± std): {test_metrics['mae']:.6f} ± {test_metrics['mae_std']:.6f}\n")
        f.write(f"PSD (mean ± std): {test_metrics['freq_diff']:.6f} ± {test_metrics['freq_diff_std']:.6f}\n")
        f.write(f"FC sim (mean ± std): {test_metrics['fc_similarity']:.6f} ± {test_metrics['fc_similarity_std']:.6f}\n")
        f.write(f"cFID-FC: {cfid_score:.6f}\n")
        f.write(f"FC-Precision@5%: {prec5:.6f}\n")
        f.write(f"Num subjects: {test_metrics['num_subjects']}\n")
    print(f"Test results written to {path}")


if __name__ == "__main__":
    main()
