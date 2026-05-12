#!/usr/bin/env python3
"""
Run STAGIN classification on Biopoint resting-state fMRI only.
No synthetic data is used.

Run from the repo root or from stagin_biopoint/:
  python stagin_biopoint/run_rest_only.py \
      --sourcedir ./data/biopoint_data \
      --csv_path  .//fMRI_autism/v3_causality_lstm/causal_lstm_2025/biopoint_data.csv \
      --targetdir ./results_stagin_rest_only \
      --k_fold 5 --max_folds 5 \
      --num_epochs 40

Results per fold are written by LoggerSTAGIN (CSV) and a summary is
printed to stdout and saved to <targetdir>/summary.txt.
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
from einops import repeat
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score

# ── path setup: works whether called from repo root or stagin_biopoint/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from model import ModelSTAGIN
from dataset_biopoint import DatasetBiopointRest
from util import bold
from util.logger import LoggerSTAGIN


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="STAGIN on Biopoint rest fMRI (real data only, no synthetic)"
    )
    # ── Data paths
    p.add_argument("--sourcedir", type=str,
                   default="./data/biopoint_data",
                   help="Biopoint data root (contains output/<subject_id>/rest/*.npy)")
    p.add_argument("--csv_path", type=str,
                   default=".//fMRI_autism/v3_causality_lstm/causal_lstm_2025/biopoint_data.csv",
                   help="CSV with columns: subject_id, group")
    p.add_argument("--ts_filename_suffix", type=str, default="_shen268_ts.npy",
                   help="Time-series file suffix (Biopoint default: _shen268_ts.npy)")
    p.add_argument("--targetdir", type=str, default="./results_stagin_rest_only",
                   help="Output directory for checkpoints, attention maps, and results")
    p.add_argument("--dynamic_length", type=int, default=None,
                   help="Crop time series to this many timepoints (None = use all)")

    # ── Cross-validation
    p.add_argument("--k_fold", type=int, default=5,
                   help="Number of CV folds. Use 1 for a single train/test split.")
    p.add_argument("--max_folds", type=int, default=5,
                   help="How many folds to actually run (≤ k_fold). "
                        "Set to 1 for a quick single-fold run.")
    p.add_argument("--train_ratio", type=float, default=0.8,
                   help="Train fraction when k_fold=1 (ignored for k_fold>1).")

    # ── Dynamic-FC / STAGIN graph settings
    p.add_argument("--window_size", type=int, default=50,
                   help="Sliding-window length (timepoints) for dynamic FC")
    p.add_argument("--window_stride", type=int, default=3,
                   help="Stride between FC window start points")

    # ── STAGIN model hyperparameters
    p.add_argument("--hidden_dim", type=int, default=128,
                   help="STAGIN hidden dimension. Try 64, 128, 256.")
    p.add_argument("--num_heads", type=int, default=1,
                   help="Number of attention heads")
    p.add_argument("--num_layers", type=int, default=4,
                   help="Number of STAGIN layers")
    p.add_argument("--sparsity", type=int, default=30,
                   help="Top-K sparsity for adjacency (30 = keep top-30 edges per node)")
    p.add_argument("--dropout", type=float, default=0.5,
                   help="Dropout rate")
    p.add_argument("--readout", type=str, default="sero",
                   choices=["garo", "sero", "mean"],
                   help="Readout function")
    p.add_argument("--cls_token", type=str, default="sum",
                   choices=["sum", "mean", "param"],
                   help="CLS token aggregation")

    # ── Optimiser hyperparameters
    p.add_argument("--num_epochs", type=int, default=40)
    p.add_argument("--minibatch_size", "-b", type=int, default=8,
                   help="Batch size")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--max_lr", type=float, default=1e-3,
                   help="Peak LR for OneCycleLR scheduler")
    p.add_argument("--reg_lambda", type=float, default=1e-5,
                   help="Orthogonality regularisation weight")
    p.add_argument("--clip_grad", type=float, default=0.0,
                   help="Gradient clipping value (0 = no clipping)")

    # ── Misc
    p.add_argument("--validate", action="store_true",
                   help="Run validation pass each epoch (prints val metrics)")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# STAGIN step helper  (identical to experiment.py — self-contained)
# ─────────────────────────────────────────────────────────────────────────────

def step(model, criterion, dyn_v, dyn_a, sampling_endpoints, t, label,
         reg_lambda, clip_grad=0.0, device="cpu", optimizer=None, scheduler=None):
    if optimizer is None:
        model.eval()
    else:
        model.train()

    logit, attention, latent, reg_ortho = model(
        dyn_v.to(device), dyn_a.to(device), t.to(device), sampling_endpoints
    )
    loss = criterion(logit, label.to(device))
    loss = loss + reg_lambda * reg_ortho

    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0.0:
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    return logit, loss, attention, latent, reg_ortho


# ─────────────────────────────────────────────────────────────────────────────
# Compute summary metrics from logger samples
# ─────────────────────────────────────────────────────────────────────────────

def _fold_metrics(samples):
    """Derive acc, f1, auc, sensitivity, specificity from logger samples dict."""
    true  = np.array(samples["true"])
    pred  = np.array(samples["pred"])
    prob  = np.array(samples["prob"])   # (N, num_classes)

    acc  = float((pred == true).mean() * 100)
    f1   = float(f1_score(true, pred, zero_division=1.0))
    try:
        auc = float(roc_auc_score(true, prob[:, 1]))
    except Exception:
        auc = 0.0
    try:
        cm = confusion_matrix(true, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except Exception:
        sensitivity = specificity = 0.0

    return dict(acc=acc, f1=f1, auc=auc, sensitivity=sensitivity, specificity=specificity)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    os.makedirs(args.targetdir, exist_ok=True)
    os.makedirs(os.path.join(args.targetdir, "model"), exist_ok=True)
    os.makedirs(os.path.join(args.targetdir, "attention"), exist_ok=True)

    # ── Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda requested but CUDA unavailable; running on CPU.")

    print(f"\nBiopoint STAGIN REST-ONLY classification")
    print(f"  sourcedir  : {args.sourcedir}")
    print(f"  csv_path   : {args.csv_path}")
    print(f"  targetdir  : {args.targetdir}")
    print(f"  device     : {device}")
    print(f"  k_fold     : {args.k_fold}  (running {args.max_folds} fold(s))")
    print(f"  epochs     : {args.num_epochs}")

    # ── Dataset (real rest-fMRI only)
    dataset = DatasetBiopointRest(
        sourcedir=args.sourcedir,
        csv_path=args.csv_path,
        k_fold=args.k_fold,
        train_ratio=args.train_ratio,
        dynamic_length=args.dynamic_length,
        ts_filename_suffix=args.ts_filename_suffix,
    )

    dynamic_length = args.dynamic_length or dataset.num_timepoints
    folds_to_run   = dataset.folds[: args.max_folds]
    print(f"  Folds to run : {folds_to_run}")

    logger   = LoggerSTAGIN(dataset.folds, dataset.num_classes)
    all_metrics = []

    criterion = torch.nn.CrossEntropyLoss()

    for k in folds_to_run:
        fold_model_dir = os.path.join(args.targetdir, "model", str(k))
        fold_attn_dir  = os.path.join(args.targetdir, "attention", str(k))
        os.makedirs(fold_model_dir, exist_ok=True)
        os.makedirs(fold_attn_dir,  exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  FOLD {k}  |  STAGIN  |  rest-fMRI only")
        print(f"{'='*60}")

        # ── Build model
        model = ModelSTAGIN(
            input_dim=dataset.num_nodes,
            hidden_dim=args.hidden_dim,
            num_classes=dataset.num_classes,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            sparsity=args.sparsity,
            dropout=args.dropout,
            cls_token=args.cls_token,
            readout=args.readout,
        ).to(device)

        # ── Build train DataLoader first so len() is exact for OneCycleLR
        dataset.set_fold(k, train=True)
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.minibatch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        steps_per_epoch = max(1, len(train_loader))

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.max_lr,
            epochs=args.num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.2,
            div_factor=args.max_lr / args.lr,
            final_div_factor=1000,
        )
        print(f"  OneCycleLR: steps_per_epoch={steps_per_epoch}  "
              f"total_steps={steps_per_epoch * args.num_epochs}")

        # ── Training loop
        dyn_v = None  # initialised on first batch
        for epoch in range(args.num_epochs):
            logger.initialize(k)
            dataset.set_fold(k, train=True)
            loss_acc = 0.0

            for i, x in enumerate(tqdm(train_loader, ncols=60,
                                        desc=f"Train k:{k} e:{epoch+1}/{args.num_epochs}")):
                dyn_a, sampling_points = bold.process_dynamic_fc(
                    x["timeseries"], args.window_size, args.window_stride, dynamic_length
                )
                sampling_endpoints = [p + args.window_size for p in sampling_points]

                # Rebuild dyn_v when snapshot count changes (first batch or different subject length)
                B_cur = dyn_a.shape[0]
                T_cur = dyn_a.shape[1]
                if dyn_v is None or dyn_v.shape[1] != T_cur or dyn_v.shape[0] != B_cur:
                    dyn_v = repeat(
                        torch.eye(dataset.num_nodes),
                        "n1 n2 -> b t n1 n2",
                        t=T_cur,
                        b=args.minibatch_size,
                    )
                if dyn_a.shape[0] < args.minibatch_size:
                    dyn_v = dyn_v[: dyn_a.shape[0]]

                t_seq = x["timeseries"].permute(1, 0, 2)
                label = x["label"]

                logit, loss, attention, latent, reg_ortho = step(
                    model=model,
                    criterion=criterion,
                    dyn_v=dyn_v,
                    dyn_a=dyn_a,
                    sampling_endpoints=sampling_endpoints,
                    t=t_seq,
                    label=label,
                    reg_lambda=args.reg_lambda,
                    clip_grad=args.clip_grad,
                    device=device,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
                pred = logit.argmax(1)
                prob = logit.softmax(1)
                loss_acc += loss.detach().cpu().item()
                logger.add(
                    k=k,
                    pred=pred.detach().cpu().numpy(),
                    true=label.detach().cpu().numpy(),
                    prob=prob.detach().cpu().numpy(),
                )

            # Reset dyn_v for next epoch (batch order may change)
            dyn_v = None

            if (epoch + 1) % 10 == 0 or epoch == 0:
                tr_metrics = logger.evaluate(k, print_metric=False)
                print(
                    f"    Epoch {epoch+1:3d}/{args.num_epochs}  "
                    f"loss={loss_acc/max(steps_per_epoch,1):.4f}  "
                    f"acc={tr_metrics.get('accuracy', 0)*100:.1f}%"
                )

            # ── Optional per-epoch validation
            if args.validate:
                logger.initialize(k)
                dataset.set_fold(k, train=False)
                val_loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=args.minibatch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=(device.type == "cuda"),
                )
                for i, x in enumerate(val_loader):
                    with torch.no_grad():
                        dyn_a, sampling_points = bold.process_dynamic_fc(
                            x["timeseries"], args.window_size, args.window_stride, dynamic_length
                        )
                        sampling_endpoints = [p + args.window_size for p in sampling_points]
                        B_cur = dyn_a.shape[0]
                        T_cur = dyn_a.shape[1]
                        dyn_v_val = repeat(
                            torch.eye(dataset.num_nodes),
                            "n1 n2 -> b t n1 n2",
                            t=T_cur, b=args.minibatch_size,
                        )
                        if dyn_a.shape[0] < args.minibatch_size:
                            dyn_v_val = dyn_v_val[: dyn_a.shape[0]]
                        t_seq = x["timeseries"].permute(1, 0, 2)
                        label = x["label"]
                        logit, loss, attention, latent, reg_ortho = step(
                            model=model, criterion=criterion,
                            dyn_v=dyn_v_val, dyn_a=dyn_a,
                            sampling_endpoints=sampling_endpoints,
                            t=t_seq, label=label,
                            reg_lambda=args.reg_lambda, clip_grad=args.clip_grad,
                            device=device, optimizer=None, scheduler=None,
                        )
                        pred = logit.argmax(1)
                        prob = logit.softmax(1)
                        logger.add(
                            k=k,
                            pred=pred.detach().cpu().numpy(),
                            true=label.detach().cpu().numpy(),
                            prob=prob.detach().cpu().numpy(),
                        )
                val_metrics = logger.evaluate(k, print_metric=False)
                print(
                    f"    Val  epoch {epoch+1:3d}:  "
                    f"acc={val_metrics.get('accuracy', 0)*100:.1f}%"
                )

        # ── Save checkpoint for this fold
        torch.save(model.state_dict(), os.path.join(fold_model_dir, "model.pth"))
        print(f"    Checkpoint saved → {fold_model_dir}/model.pth")

        # ── Test evaluation on held-out fold
        print(f"\n  [Fold {k}] Test evaluation...")
        logger.initialize(k)
        dataset.set_fold(k, train=False)
        latent_acc = []
        fold_attn  = {"node_attention": [], "time_attention": []}
        dyn_v_test = None

        # Test dataloader: batch_size=1 for reliable attention saving (matches original)
        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        )
        for i, x in enumerate(tqdm(test_loader, ncols=60, desc=f"Test k:{k}")):
            with torch.no_grad():
                dyn_a, sampling_points = bold.process_dynamic_fc(
                    x["timeseries"], args.window_size, args.window_stride, dynamic_length
                )
                sampling_endpoints = [p + args.window_size for p in sampling_points]
                T_cur = dyn_a.shape[1]
                dyn_v_test = repeat(
                    torch.eye(dataset.num_nodes),
                    "n1 n2 -> b t n1 n2",
                    t=T_cur, b=1,
                )
                t_seq = x["timeseries"].permute(1, 0, 2)
                label = x["label"]
                logit, loss, attention, latent, reg_ortho = step(
                    model=model, criterion=criterion,
                    dyn_v=dyn_v_test, dyn_a=dyn_a,
                    sampling_endpoints=sampling_endpoints,
                    t=t_seq, label=label,
                    reg_lambda=args.reg_lambda, clip_grad=args.clip_grad,
                    device=device, optimizer=None, scheduler=None,
                )
                pred = logit.argmax(1)
                prob = logit.softmax(1)
                logger.add(
                    k=k,
                    pred=pred.detach().cpu().numpy(),
                    true=label.detach().cpu().numpy(),
                    prob=prob.detach().cpu().numpy(),
                )
                latent_acc.append(latent.detach().cpu().numpy())
                fold_attn["node_attention"].append(attention["node-attention"].detach().cpu().numpy())
                fold_attn["time_attention"].append(attention["time-attention"].detach().cpu().numpy())

        # ── Save attention and latent
        for key, value in fold_attn.items():
            torch.save(value, os.path.join(fold_attn_dir, f"{key}.pth"))
        np.save(os.path.join(fold_attn_dir, "latent.npy"), np.concatenate(latent_acc))

        # ── Per-fold metrics
        samples = logger.get(k)
        fm = _fold_metrics(samples)
        all_metrics.append(fm)
        print(
            f"  [Fold {k}] TEST → "
            f"acc={fm['acc']:.2f}%  f1={fm['f1']:.4f}  "
            f"auc={fm['auc']:.4f}  sens={fm['sensitivity']:.4f}  spec={fm['specificity']:.4f}"
        )

        # ── Write per-fold CSV (matches LoggerSTAGIN interface)
        logger.to_csv(args.targetdir, k)

    # ── Aggregate across folds
    if not all_metrics:
        print("No folds completed.")
        return

    mean_acc  = np.mean([m["acc"]  for m in all_metrics])
    mean_f1   = np.mean([m["f1"]   for m in all_metrics])
    mean_auc  = np.mean([m["auc"]  for m in all_metrics])
    mean_sens = np.mean([m["sensitivity"] for m in all_metrics])
    mean_spec = np.mean([m["specificity"] for m in all_metrics])
    std_acc   = np.std( [m["acc"]  for m in all_metrics])
    std_f1    = np.std( [m["f1"]   for m in all_metrics])
    std_auc   = np.std( [m["auc"]  for m in all_metrics])
    std_sens  = np.std( [m["sensitivity"] for m in all_metrics])
    std_spec  = np.std( [m["specificity"] for m in all_metrics])

    print(f"\n{'='*60}")
    print(f"  SUMMARY  —  STAGIN rest-only  ({len(all_metrics)}/{args.k_fold} fold(s))")
    print(f"{'='*60}")
    print(f"  Acc         : {mean_acc:.2f} ± {std_acc:.2f} %")
    print(f"  F1          : {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"  AUC         : {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  Sensitivity : {mean_sens:.4f} ± {std_sens:.4f}")
    print(f"  Specificity : {mean_spec:.4f} ± {std_spec:.4f}")

    # ── Write summary.txt
    summary_path = os.path.join(args.targetdir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"STAGIN REST-ONLY — Biopoint — {len(all_metrics)}/{args.k_fold} fold(s)\n")
        f.write(
            f"epochs={args.num_epochs}  hidden_dim={args.hidden_dim}  "
            f"num_heads={args.num_heads}  num_layers={args.num_layers}  "
            f"sparsity={args.sparsity}  window_size={args.window_size}\n\n"
        )
        f.write(f"acc_mean\t{mean_acc:.4f}\nacc_std\t{std_acc:.4f}\n")
        f.write(f"f1_mean\t{mean_f1:.4f}\nf1_std\t{std_f1:.4f}\n")
        f.write(f"auc_mean\t{mean_auc:.4f}\nauc_std\t{std_auc:.4f}\n")
        f.write(f"sensitivity_mean\t{mean_sens:.4f}\nsensitivity_std\t{std_sens:.4f}\n")
        f.write(f"specificity_mean\t{mean_spec:.4f}\nspecificity_std\t{std_spec:.4f}\n")
        f.write("\n")
        for i, m in enumerate(all_metrics):
            f.write(
                f"fold_{i}\t"
                f"acc={m['acc']:.4f}\tf1={m['f1']:.4f}\t"
                f"auc={m['auc']:.4f}\tsensitivity={m['sensitivity']:.4f}\t"
                f"specificity={m['specificity']:.4f}\n"
            )
    print(f"\nSummary written → {summary_path}")

    # ── Full logger CSV (all folds combined)
    logger.to_csv(args.targetdir)


if __name__ == "__main__":
    main()
