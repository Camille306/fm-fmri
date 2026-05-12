#!/usr/bin/env python3
"""
Re-evaluate all best baseline (FM-TS) checkpoints with a unified set of metrics.

For each checkpoint: load model, run inference on test windows to get paired
(real, generated) task series, then compute:
  - MAE (mean absolute error, window-level average)
  - PSD (mean absolute PSD difference, window-level average)
  - FC similarity (Pearson on upper-triangle FC, window-level average)
  - FC top 5% precision (window-level average)
  - cFID-FC (conditional Fréchet distance on FC features)
  - Discriminative score (LSTM classifier |val_acc - 0.5|)

Config: JSON array of { "name": "baseline_name", "load_dir": "/path/to/checkpoint" [, "task_name": "emotion" ] }.
task_name is read from each entry (build_baselines_config.py fills it); --task_name is only a fallback when an entry omits it.

Usage:
    python re_eval/run_all_baselines_metrics.py \
        --config re_eval/baselines_config.json \
        --out_csv re_eval/results_baselines.tsv
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Ensure fm-fmri is on path and load metric functions (no model yet)
REPO_ROOT = Path(__file__).resolve().parent.parent
FM_FMRI_DIR = REPO_ROOT / "fm-fmri"
sys.path.insert(0, str(FM_FMRI_DIR))


def _import_fm_fmri():
    import importlib.util
    spec = importlib.util.spec_from_file_location("fm_fmri", FM_FMRI_DIR / "fm_fmri.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fm_fmri"] = mod
    spec.loader.exec_module(mod)
    return mod


# Import re_eval modules (same package as this script)
from run_discriminative_score import load_fmts_and_dataset, collect_real_and_generated
from lstm_classifier import LSTMTimeSeriesClassifier
from fc_utils import cfid_fc


def compute_window_level_metrics(X_real, X_gen, fm_fmri_module, fs=0.72):
    """
    X_real, X_gen: (N, T, V). Compute per-window MAE, PSD, FC similarity, FC precision@5; return means.
    """
    compute_frequency_difference = fm_fmri_module.compute_frequency_difference
    compute_fc_similarity = fm_fmri_module.compute_fc_similarity
    compute_fc_topk_precision_recall_auc = fm_fmri_module.compute_fc_topk_precision_recall_auc

    N = X_real.shape[0]
    mae_list = []
    psd_list = []
    fc_sim_list = []
    prec5_list = []

    for i in range(N):
        real_i = X_real[i]   # (T, V)
        gen_i = X_gen[i]     # (T, V)
        mae_list.append(float(np.mean(np.abs(real_i - gen_i))))
        psd_list.append(compute_frequency_difference(gen_i, real_i, fs=fs))
        fc_sim_list.append(compute_fc_similarity(gen_i, real_i))
        topk = compute_fc_topk_precision_recall_auc(gen_i, real_i, k_percentiles=(5,))
        prec5_list.append(topk.get("precision_at_5", float("nan")))

    return {
        "mae": float(np.mean(mae_list)),
        "psd": float(np.mean(psd_list)),
        "fc_similarity": float(np.mean(fc_sim_list)),
        "fc_precision_at_5": float(np.nanmean(prec5_list)),
    }


def compute_discriminative_score(X_real, X_gen, V, device, classifier_epochs=30, classifier_val_ratio=0.2,
                                seed=42, classifier_lr=1e-3, classifier_hidden=128, classifier_layers=2,
                                classifier_dropout=0.2, debug_scale=False):
    """
    Train LSTM classifier on mixed real/generated and return |val_acc - 0.5|.

    No data leakage: X_real and X_gen are from the test set only (model never trained on them).
    We form 2N samples (N real, N generated), shuffle with fixed seed, then take a disjoint
    train/val split (val = first n_val after shuffle, train = rest). So no sample appears
    in both train and val.
    """
    n = X_real.shape[0]
    X = np.concatenate([X_real, X_gen], axis=0).astype(np.float32)
    y = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)], axis=0)

    # Reproducible shuffle so train/val split is deterministic for this seed
    np.random.seed(seed)
    perm = np.random.permutation(len(y))
    X, y = X[perm], y[perm]

    n_val = int(len(y) * classifier_val_ratio)
    X_val, y_val = X[:n_val], y[:n_val]
    X_train, y_train = X[n_val:], y[n_val:]
    # Sanity: train and val are disjoint and cover all 2N samples
    assert n_val + len(y_train) == len(y) and n_val > 0 and len(y_train) > 0

    if debug_scale:
        r_mean, r_std = float(np.mean(X_real)), float(np.std(X_real))
        g_mean, g_std = float(np.mean(X_gen)), float(np.std(X_gen))
        print(f"        [scale check] real mean={r_mean:.4f} std={r_std:.4f}  gen mean={g_mean:.4f} std={g_std:.4f}", flush=True)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=min(64, len(train_ds)), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=min(64, len(val_ds)), shuffle=False, num_workers=0)

    clf = LSTMTimeSeriesClassifier(
        input_size=V,
        hidden_size=classifier_hidden,
        num_layers=classifier_layers,
        dropout=classifier_dropout,
        bidirectional=False,
    ).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=classifier_lr)
    criterion = torch.nn.CrossEntropyLoss()

    torch.manual_seed(seed)
    for _ in range(classifier_epochs):
        clf.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(clf(xb), yb)
            loss.backward()
            opt.step()
        clf.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = clf(xb).argmax(dim=1)
            val_correct += (pred == yb).sum().item()
            val_total += yb.size(0)
    val_acc = val_correct / val_total if val_total else 0.5
    return abs(val_acc - 0.5)


def main():
    p = argparse.ArgumentParser(description="Re-evaluate all baseline checkpoints: MAE, PSD, FC sim, FC top5% prec, cFID-FC, discriminative score")
    p.add_argument("--config", type=str, required=True, help="JSON file: array of { name, load_dir [, task_name ] }")
    p.add_argument("--data_root", type=str, default=os.getenv("DATA_ROOT", "./data/hcp-resting-fc"),
                   help="HCP resting data root (default: HCP path or DATA_ROOT)")
    p.add_argument("--task_root", type=str, default=os.getenv("TASK_ROOT", "./data/hcp-task-ts"),
                   help="HCP task data root (default: HCP path or TASK_ROOT)")
    p.add_argument("--task_name", type=str, default="emotion",
                   help="Fallback task name when a config entry has no task_name (default: emotion)")
    p.add_argument("--use_evs", action="store_true")
    p.add_argument("--ev_root", type=str, default=None)
    p.add_argument("--lookback_length", type=int, default=512)
    p.add_argument("--prediction_length", type=int, default=None)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--ode_steps", type=int, default=50)
    p.add_argument("--classifier_epochs", type=int, default=30)
    p.add_argument("--classifier_val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_fc_dim", type=int, default=500,
                   help="cFID-FC random projection dim (default: 500; set 0 to use full d=V(V-1)/2)")
    p.add_argument("--out_csv", type=str, default=None, help="Output CSV path (default: print only)")
    p.add_argument("--debug_scale", action="store_true", help="Print mean/std of real vs generated before discriminative score (check for scale mismatch)")
    # FMTS architecture defaults (overridden by checkpoint)
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

    args = p.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config, "r") as f:
        config = json.load(f)
    if not isinstance(config, list):
        config = [config]

    print("=" * 70)
    print("Re-evaluation: unified metrics (MAE, PSD, FC sim, FC prec@5, cFID-FC, discriminative)")
    print("=" * 70)
    print(f"Config: {args.config}  |  Entries: {len(config)}  |  Device: {args.device}")
    print()

    fm_fmri = _import_fm_fmri()

    rows = []
    for idx, entry in enumerate(config):
        name = entry.get("name", f"baseline_{idx}")
        model_type = entry.get("model_type", "fmts")
        if model_type != "fmts":
            print(f"[Skip] {name}: model_type={model_type} (only fmts supported)")
            continue
        load_dir = entry.get("load_dir")
        if not load_dir:
            print(f"[Skip] {name}: no load_dir")
            continue
        task_name = entry.get("task_name", args.task_name)
        # Infer use_evs from name if not set in JSON: "no_ev" -> False, "with_ev" -> True, else CLI default
        if "use_evs" in entry:
            use_evs = bool(entry["use_evs"])
        elif "no_ev" in name:
            use_evs = False
        elif "with_ev" in name:
            use_evs = True
        else:
            use_evs = args.use_evs
        args_copy = argparse.Namespace(**vars(args))
        args_copy.load_dir = load_dir
        args_copy.task_name = task_name
        args_copy.use_evs = use_evs

        print("\n" + "-" * 70)
        print(f"Baseline [{idx+1}/{len(config)}]  {name}")
        print(f"  load_dir={load_dir}  task_name={task_name}  use_evs={use_evs}")
        print("-" * 70)

        try:
            print("  [1/5] Loading model and test dataset...", flush=True)
            model, test_loader, pred_len, V, args_copy = load_fmts_and_dataset(args_copy)
            n_windows = len(test_loader.dataset)
            print(f"        Done. Test windows: {n_windows}, T={pred_len}, V={V}", flush=True)

            print("  [2/5] Running inference (collecting real & generated pairs)...", flush=True)
            X_real, X_gen = collect_real_and_generated(
                model, test_loader, pred_len, args_copy.device, ode_steps=args_copy.ode_steps
            )
            n = X_real.shape[0]
            print(f"        Done. Collected {n} paired windows.", flush=True)

        except Exception as e:
            print(f"  Error at load/inference: {e}", flush=True)
            rows.append({
                "name": name,
                "load_dir": load_dir,
                "mae": float("nan"), "psd": float("nan"), "fc_similarity": float("nan"),
                "fc_precision_at_5": float("nan"), "cfid_fc": float("nan"), "discriminative_score": float("nan"),
            })
            continue

        print("  [3/5] Computing window-level metrics (MAE, PSD, FC similarity, FC prec@5)...", flush=True)
        win_metrics = compute_window_level_metrics(X_real, X_gen, fm_fmri)
        print(f"        Done. MAE={win_metrics['mae']:.6f}  PSD={win_metrics['psd']:.6f}  FC_sim={win_metrics['fc_similarity']:.4f}  FC_prec@5={win_metrics['fc_precision_at_5']:.4f}", flush=True)

        print("  [4/5] Computing cFID-FC...", flush=True)
        rng = np.random.default_rng(args.seed)
        max_fc_dim = getattr(args, "max_fc_dim", 500)
        if max_fc_dim <= 0:
            max_fc_dim = None
        cfid = cfid_fc(X_real, X_gen, max_fc_dim=max_fc_dim, rng=rng)
        if max_fc_dim is not None:
            print(f"        (max_fc_dim={max_fc_dim})", flush=True)
        print(f"        Done. cFID-FC={cfid:.6f}", flush=True)

        print("  [5/5] Computing discriminative score (LSTM classifier)...", flush=True)
        discr = compute_discriminative_score(
            X_real, X_gen, V, args_copy.device,
            classifier_epochs=args.classifier_epochs,
            classifier_val_ratio=args.classifier_val_ratio,
            seed=args.seed,
            debug_scale=getattr(args, "debug_scale", False),
        )
        print(f"        Done. Discriminative score={discr:.4f}", flush=True)

        row = {
            "name": name,
            "load_dir": load_dir,
            "mae": win_metrics["mae"],
            "psd": win_metrics["psd"],
            "fc_similarity": win_metrics["fc_similarity"],
            "fc_precision_at_5": win_metrics["fc_precision_at_5"],
            "cfid_fc": cfid,
            "discriminative_score": discr,
        }
        rows.append(row)
        print(f"  >>> Baseline [{idx+1}/{len(config)}] complete: {name}", flush=True)

    # Print table
    print("\n" + "=" * 100)
    print("Re-evaluation finished. Summary")
    print("=" * 100)
    col_order = ["name", "mae", "psd", "fc_similarity", "fc_precision_at_5", "cfid_fc", "discriminative_score"]
    for r in rows:
        print("\t".join(str(r.get(c, "")) for c in col_order))

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write("\t".join(col_order) + "\n")
            for r in rows:
                f.write("\t".join(str(r.get(c, "")) for c in col_order) + "\n")
        print(f"\nSaved results to {out_path}")

    return rows


if __name__ == "__main__":
    main()
