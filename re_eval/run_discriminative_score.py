#!/usr/bin/env python3
"""
Discriminative score pipeline for rest-to-task fMRI generation.

Workflow:
1. Load the best trained fm-fmri model and run inference on test windows to get generated task time series.
2. Mix real task time series (from test set) with generated task time series.
3. Train an LSTM binary classifier (real=0, generated=1) on the mixed data.
4. Discriminative score = |accuracy - 0.5| (0 = indistinguishable, 0.5 = perfectly distinguishable).

Usage:
    python re_eval/run_discriminative_score.py \
        --load_dir /path/to/best/model \
        --data_root /path/to/hcp-resting-fc \
        --task_root /path/to/hcp-task-ts \
        --task_name emotion
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Add fm-fmri so we can import dataset and model
REPO_ROOT = Path(__file__).resolve().parent.parent
FM_FMRI_DIR = REPO_ROOT / "fm-fmri"
sys.path.insert(0, str(FM_FMRI_DIR))

from dataset import HCPRestingFCDataset
from lstm_classifier import LSTMTimeSeriesClassifier


# We need FMRIWindowDataset and FMTS from fm_fmri.py
def _import_fm_fmri():
    import importlib.util
    spec = importlib.util.spec_from_file_location("fm_fmri", FM_FMRI_DIR / "fm_fmri.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fm_fmri"] = mod
    spec.loader.exec_module(mod)
    return mod


def get_test_loader_only(args):
    """Build HCP dataset and test DataLoader only (no model). Returns (test_loader, pred_len, V, args)."""
    fm_fmri = _import_fm_fmri()
    FMRIWindowDataset = fm_fmri.FMRIWindowDataset

    ds = HCPRestingFCDataset(
        data_root=str(args.data_root),
        task_root=str(args.task_root) if args.task_root else None,
        task_name=args.task_name,
        use_evs=args.use_evs,
        ev_root=str(args.ev_root) if args.ev_root else None,
    )
    if len(ds) == 0:
        raise SystemExit("No subjects in dataset. Check data_root, task_root, task_name.")

    train_ratio = getattr(args, "train_ratio", 0.7)
    val_ratio = getattr(args, "val_ratio", 0.15)
    lookback = getattr(args, "lookback_length", 512)
    pred_len = getattr(args, "prediction_length", None)
    stride = getattr(args, "stride", 10)

    # Infer prediction_length from task if not set
    if pred_len is None:
        sample_sid = ds.subject_ids[0]
        task_ts = ds.load_task_subject(sample_sid)
        if task_ts.ndim == 1:
            task_ts = task_ts.reshape(-1, 1)
        pred_len = task_ts.shape[0]
        args.prediction_length = pred_len

    train_ds = FMRIWindowDataset(
        dataset=ds,
        lookback_length=lookback,
        prediction_length=pred_len,
        stride=stride,
        normalize=True,
        split="train",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    val_ds = FMRIWindowDataset(
        dataset=ds,
        lookback_length=lookback,
        prediction_length=pred_len,
        stride=stride,
        normalize=True,
        split="val",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    test_ds = FMRIWindowDataset(
        dataset=ds,
        lookback_length=lookback,
        prediction_length=pred_len,
        stride=stride,
        normalize=True,
        split="test",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    test_ds.rest_means, test_ds.rest_stds = train_ds.rest_means, train_ds.rest_stds
    test_ds.task_means, test_ds.task_stds = train_ds.task_means, train_ds.task_stds

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Get V and pred_len from a sample
    sample = next(iter(test_loader))
    V = int(sample["input"].shape[-1])
    pred_len = int(sample["target"].shape[1])
    return test_loader, pred_len, V, args


def load_fmts_and_dataset(args):
    """Build HCP dataset, windowed test dataset, load checkpoint, build FMTS model."""
    fm_fmri = _import_fm_fmri()
    FMTS = fm_fmri.FMTS
    test_loader, pred_len, V, args = get_test_loader_only(args)

    # Load checkpoint and build FMTS
    best_path = os.path.join(args.load_dir, "best_fmts.pth")
    if not os.path.isfile(best_path):
        raise SystemExit(f"Checkpoint not found: {best_path}")

    ckpt = torch.load(best_path, map_location=args.device)
    saved_args = ckpt.get("args")
    if saved_args is not None:
        for key in [
            "use_evs", "use_hrf_kernel", "use_ev_hrf_timecourse",
            "rest_encoder", "rest_hidden", "ctx_dim", "t_dim",
            "rest_patch_len", "rest_num_layers", "rest_nhead",
            "rest_dim_feedforward", "prior_K", "use_prior_detach",
            "num_conditions", "d_ev", "hrf_kernel_len", "hrf_num_basis",
            "hrf_per_roi", "ev_hrf_kernel_len", "ev_hrf_num_basis",
            "no_ev_hrf_delay_width", "ev_hrf_smooth_boxcar", "ev_hrf_boxcar_sigma",
        ]:
            if key in saved_args and hasattr(args, key):
                setattr(args, key, saved_args[key])

    model = FMTS(
        v_dim=V,
        rest_hidden=args.rest_hidden,
        ctx_dim=args.ctx_dim,
        t_dim=args.t_dim,
        rest_encoder=args.rest_encoder,
        rest_patch_len=args.rest_patch_len,
        rest_num_layers=args.rest_num_layers,
        rest_nhead=args.rest_nhead,
        rest_dim_feedforward=args.rest_dim_feedforward,
        use_evs=args.use_evs,
        num_conditions=args.num_conditions,
        d_ev=args.d_ev,
        use_hrf_kernel=args.use_hrf_kernel,
        hrf_kernel_len=args.hrf_kernel_len,
        hrf_num_basis=args.hrf_num_basis,
        hrf_per_roi=args.hrf_per_roi,
        use_ev_hrf_timecourse=args.use_ev_hrf_timecourse,
        ev_hrf_kernel_len=args.ev_hrf_kernel_len,
        ev_hrf_num_basis=args.ev_hrf_num_basis,
        ev_hrf_use_delay_width=not getattr(args, "no_ev_hrf_delay_width", False),
        ev_hrf_smooth_boxcar=args.ev_hrf_smooth_boxcar,
        ev_hrf_boxcar_sigma=args.ev_hrf_boxcar_sigma,
        prior_K=args.prior_K,
        use_prior_detach=args.use_prior_detach,
    ).to(args.device)

    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    return model, test_loader, pred_len, V, args


def collect_real_and_generated(model, test_loader, pred_len, device, ode_steps=50):
    """Run inference and collect real (target) and generated (model.sample) task series."""
    real_list = []
    gen_list = []

    with torch.no_grad():
        for batch in test_loader:
            x_rest = batch["input"].to(device).float()
            x_task = batch["target"].to(device).float()
            starts = batch["task_start_idx"]
            ev = batch["ev"].to(device).float() if batch.get("ev") is not None else None
            ev_mask = batch["ev_mask"].to(device).float() if batch.get("ev_mask") is not None else None

            if isinstance(starts, torch.Tensor):
                task_start_idx = starts.to(device)
            else:
                task_start_idx = torch.tensor(starts, device=device, dtype=torch.float32)

            x_pred = model.sample(
                x_rest, T_pred=pred_len, steps=ode_steps,
                ev=ev, ev_mask=ev_mask, task_start_idx=task_start_idx,
            )

            real_list.append(x_task.cpu().numpy())
            gen_list.append(x_pred.cpu().numpy())

    X_real = np.concatenate(real_list, axis=0)   # (N, T, V)
    X_gen = np.concatenate(gen_list, axis=0)     # (N, T, V)
    return X_real, X_gen


def main():
    p = argparse.ArgumentParser(description="Discriminative score: LSTM classifier on real vs generated task series")
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
    # Model (for loading FMTS)
    p.add_argument("--load_dir", type=str, required=True, help="Path to directory containing best_fmts.pth")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--ode_steps", type=int, default=50)
    # FMTS architecture defaults (overridden by checkpoint if present)
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
    # LSTM classifier
    p.add_argument("--classifier_epochs", type=int, default=30)
    p.add_argument("--classifier_lr", type=float, default=1e-3)
    p.add_argument("--classifier_hidden", type=int, default=128)
    p.add_argument("--classifier_layers", type=int, default=2)
    p.add_argument("--classifier_dropout", type=float, default=0.2)
    p.add_argument("--classifier_val_ratio", type=float, default=0.2, help="Fraction of mixed data for classifier val")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug_scale", action="store_true", help="Print mean/std of real vs generated (check for scale mismatch if score is 0.5)")
    p.add_argument("--save_dir", type=str, default=None, help="Optional: save score and classifier state here")

    args = p.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Loading FM model and test data...")
    model, test_loader, pred_len, V, args = load_fmts_and_dataset(args)
    print(f"Test windows: {len(test_loader.dataset)}, T={pred_len}, V={V}")

    print("Running inference to collect generated task series...")
    X_real, X_gen = collect_real_and_generated(
        model, test_loader, pred_len, args.device, ode_steps=args.ode_steps
    )
    n = X_real.shape[0]
    print(f"Collected {n} real and {n} generated windows. Shape each: (T={pred_len}, V={V})")
    if getattr(args, "debug_scale", False):
        r_mean, r_std = float(np.mean(X_real)), float(np.std(X_real))
        g_mean, g_std = float(np.mean(X_gen)), float(np.std(X_gen))
        print(f"  [scale check] real mean={r_mean:.4f} std={r_std:.4f}  gen mean={g_mean:.4f} std={g_std:.4f}")

    # Mix: X = [real; generated], y = [0,...,0, 1,...,1]
    # No leakage: real and generated are from test set only; LSTM train/val will be disjoint random splits of this mix.
    X = np.concatenate([X_real, X_gen], axis=0).astype(np.float32)
    y = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)], axis=0)

    # Reproducible shuffle, then disjoint train/val split (val = first n_val, train = rest)
    np.random.seed(args.seed)
    perm = np.random.permutation(len(y))
    X, y = X[perm], y[perm]
    n_val = int(len(y) * args.classifier_val_ratio)
    X_val, y_val = X[:n_val], y[:n_val]
    X_train, y_train = X[n_val:], y[n_val:]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=min(64, len(train_ds)), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=min(64, len(val_ds)), shuffle=False, num_workers=0)

    # LSTM classifier
    clf = LSTMTimeSeriesClassifier(
        input_size=V,
        hidden_size=args.classifier_hidden,
        num_layers=args.classifier_layers,
        dropout=args.classifier_dropout,
        bidirectional=False,
    ).to(args.device)
    opt = torch.optim.Adam(clf.parameters(), lr=args.classifier_lr)
    criterion = torch.nn.CrossEntropyLoss()

    for ep in range(1, args.classifier_epochs + 1):
        clf.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(args.device), yb.to(args.device)
            opt.zero_grad()
            logits = clf(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item()
            pred = logits.argmax(dim=1)
            train_correct += (pred == yb).sum().item()
            train_total += yb.size(0)
        train_acc = train_correct / train_total

        clf.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(args.device), yb.to(args.device)
                logits = clf(xb)
                pred = logits.argmax(dim=1)
                val_correct += (pred == yb).sum().item()
                val_total += yb.size(0)
        val_acc = val_correct / val_total if val_total else 0.0

        if ep % 5 == 0 or ep == 1:
            print(f"  Epoch {ep}/{args.classifier_epochs}  train_loss={train_loss/len(train_loader):.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

    # Final discriminative score = |accuracy - 0.5|
    discriminative_score = abs(val_acc - 0.5)
    print()
    print("=" * 60)
    print("Discriminative score result")
    print("=" * 60)
    print(f"Classifier validation accuracy: {val_acc:.4f}")
    print(f"Discriminative score (|accuracy - 0.5|): {discriminative_score:.4f}")
    print("  (0 = generated indistinguishable from real; 0.5 = perfectly distinguishable)")
    print("=" * 60)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        out_path = Path(args.save_dir) / "discriminative_score.txt"
        with open(out_path, "w") as f:
            f.write(f"validation_accuracy\t{val_acc:.6f}\n")
            f.write(f"discriminative_score\t{discriminative_score:.6f}\n")
        print(f"Saved results to {out_path}")
        torch.save(clf.state_dict(), Path(args.save_dir) / "lstm_classifier.pth")

    return discriminative_score


if __name__ == "__main__":
    main()
