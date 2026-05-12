#!/usr/bin/env python3
"""
Run GCN and GAT classification on Biopoint resting-state fMRI only.
No synthetic data is used.  Both models are trained and evaluated
sequentially and results are written to --targetdir.

Run from the repo root or from gcn_gat_biopoint/:
  python gcn_gat_biopoint/run_rest_only.py \
      --sourcedir ./data_pi_lab/user/project/biopoint_data \
      --csv_path  ./data_pi_lab/user/fMRI_autism/v3_causality_lstm/causal_lstm_2025/biopoint_data.csv \
      --targetdir ./results_rest_only \
      --k_fold 5 --max_folds 5 \
      --num_epochs 50 --minibatch_size 4

Results per model are saved to:
  <targetdir>/gat_rest/test_results.txt
  <targetdir>/gcn_rest/test_results.txt
A combined summary is printed to stdout and written to:
  <targetdir>/summary.txt
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

# ── path setup so the script works whether called from repo root or from
#    gcn_gat_biopoint/ directly
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from model import build_model
from dataset_biopoint import DatasetBiopointRest


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="GCN + GAT on Biopoint rest fMRI (real data only, no synthetic)"
    )
    # ── Data paths
    p.add_argument("--sourcedir", type=str,
                   default="./data_pi_lab/user/project/biopoint_data",
                   help="Biopoint data root (contains output/<subject_id>/rest/*.npy)")
    p.add_argument("--csv_path", type=str,
                   default="./data_pi_lab/user/fMRI_autism/v3_causality_lstm/causal_lstm_2025/biopoint_data.csv",
                   help="CSV with columns: subject_id, group")
    p.add_argument("--ts_filename_suffix", type=str, default="_shen268_ts.npy",
                   help="Time-series file suffix (Biopoint default: _shen268_ts.npy)")
    p.add_argument("--targetdir", type=str, default="./results_rest_only",
                   help="Output directory for checkpoints and results")

    # ── Cross-validation
    p.add_argument("--k_fold", type=int, default=5,
                   help="Number of CV folds. Use 1 for a single train/test split.")
    p.add_argument("--max_folds", type=int, default=5,
                   help="How many folds to actually run (≤ k_fold). "
                        "Set to 1 for a quick single-fold run.")
    p.add_argument("--train_ratio", type=float, default=0.8,
                   help="Train fraction when k_fold=1 (ignored for k_fold>1).")

    # ── Dynamic-FC graph settings
    p.add_argument("--window_size", type=int, default=50,
                   help="Sliding-window length (timepoints) for each FC snapshot")
    p.add_argument("--window_stride", type=int, default=3,
                   help="Stride between FC snapshot start points")
    p.add_argument("--window_num", type=int, default=12,
                   help="Number of FC snapshots per subject")
    p.add_argument("--dynamic_length", type=int, default=None,
                   help="Crop time series to this many timepoints (None = use all)")
    p.add_argument("--top_k_edges", type=int, default=50,
                   help="Keep only top-K strongest |FC| edges per node per snapshot. "
                        "Reduces memory; 30-60 is a good range for 268 ROIs.")

    # ── Model hyperparameters
    p.add_argument("--hidden_dim", type=int, default=128,
                   help="GNN hidden dimension. Try 64, 128, 256.")
    p.add_argument("--dropout", type=float, default=0.2,
                   help="Dropout rate. Try 0.1, 0.2, 0.5.")

    # ── Optimiser hyperparameters
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--minibatch_size", "-b", type=int, default=4,
                   help="Batch size. Keep ≤4 for 268-ROI graphs to avoid OOM.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--scheduler_step", type=int, default=10,
                   help="StepLR step size (epochs)")
    p.add_argument("--scheduler_gamma", type=float, default=0.4,
                   help="StepLR decay factor")

    # ── Models to run
    p.add_argument("--models", type=str, nargs="+", default=["gat", "gcn"],
                   choices=["gat", "gcn"],
                   help="Which model(s) to run. Default: both gat and gcn.")

    # ── Misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Train / evaluate helpers  (identical to experiment.py — self-contained here)
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_all, correct, n = 0.0, 0, 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.squeeze(-1))
        loss.backward()
        optimizer.step()
        pred = out.argmax(dim=1)
        correct += (pred == data.y.squeeze(-1)).sum().item()
        loss_all += loss.item() * data.num_graphs
        n += data.num_graphs
    return loss_all / max(n, 1), (correct / max(n, 1)) * 100


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_all, correct, n = 0.0, 0, 0
    out_list, pred_list, y_list = [], [], []
    for data in loader:
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data.y.squeeze(-1))
        pred = out.argmax(dim=1)
        correct += (pred == data.y.squeeze(-1)).sum().item()
        loss_all += loss.item() * data.num_graphs
        n += data.num_graphs
        out_list.append(out.cpu())
        pred_list.append(pred.cpu())
        y_list.append(data.y.squeeze(-1).cpu())

    if not out_list:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    out_all  = torch.cat(out_list,  dim=0)
    pred_all = torch.cat(pred_list, dim=0)
    y_all    = torch.cat(y_list,    dim=0)

    acc = (correct / max(n, 1)) * 100
    f1  = f1_score(y_all.numpy(), pred_all.numpy(), zero_division=1.0)
    try:
        auc = roc_auc_score(y_all.numpy(), out_all[:, 1].numpy())
    except Exception:
        auc = 0.0
    try:
        cm = confusion_matrix(y_all.numpy(), pred_all.numpy(), labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except Exception:
        sensitivity = specificity = 0.0

    return loss_all / max(n, 1), acc, f1, auc, sensitivity, specificity


# ─────────────────────────────────────────────────────────────────────────────
# Run one model (train + test across folds)
# ─────────────────────────────────────────────────────────────────────────────

def run_model(model_type: str, args, device: torch.device) -> dict:
    """Train and test `model_type` on rest-only data. Returns mean metrics dict."""

    model_dir = os.path.join(args.targetdir, f"{model_type}_rest")
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Model: {model_type.upper()}  |  real rest-fMRI only  |  device: {device}")
    print(f"{'='*60}")

    # ── Build dataset (once; set_fold() selects train/test split per fold)
    dataset = DatasetBiopointRest(
        sourcedir=args.sourcedir,
        csv_path=args.csv_path,
        k_fold=args.k_fold,
        train_ratio=args.train_ratio,
        window_size=args.window_size,
        window_stride=args.window_stride,
        window_num=args.window_num,
        dynamic_length=args.dynamic_length,
        ts_filename_suffix=args.ts_filename_suffix,
        top_k_edges=args.top_k_edges,
    )

    roi_num     = dataset.num_nodes
    window_num  = dataset.window_num
    num_classes = dataset.num_classes
    criterion   = torch.nn.CrossEntropyLoss()

    folds_to_run = dataset.folds[: args.max_folds]
    print(f"  Folds to run: {folds_to_run}  "
          f"(k_fold={args.k_fold}, max_folds={args.max_folds})")

    all_metrics = []

    for k in folds_to_run:
        fold_dir = os.path.join(model_dir, "model", str(k))
        os.makedirs(fold_dir, exist_ok=True)
        ckpt_path = os.path.join(fold_dir, "model.pth")

        # ── Training
        dataset.set_fold(k, train=True)
        train_loader = DataLoader(
            dataset,
            batch_size=args.minibatch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,   # prevent singleton-batch BatchNorm crash
        )

        model = build_model(
            model_type=model_type,
            roi_num=roi_num,
            window_num=window_num,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            dropout=args.dropout,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma
        )

        print(f"\n  --- Fold {k} | train n={len(dataset)} ---")
        for epoch in range(args.num_epochs):
            tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"    Epoch {epoch+1:3d}/{args.num_epochs}  "
                    f"loss={tr_loss:.4f}  acc={tr_acc:.1f}%"
                )

        torch.save(model.state_dict(), ckpt_path)
        print(f"    Checkpoint saved → {ckpt_path}")

        # ── Testing on held-out fold
        dataset.set_fold(k, train=False)
        test_loader = DataLoader(
            dataset, batch_size=args.minibatch_size, shuffle=False, num_workers=0
        )
        test_loss, test_acc, test_f1, test_auc, test_sens, test_spec = evaluate(
            model, test_loader, criterion, device
        )
        all_metrics.append({
            "loss": test_loss, "acc": test_acc, "f1": test_f1,
            "auc": test_auc, "sensitivity": test_sens, "specificity": test_spec,
        })
        print(
            f"    Fold {k} TEST → "
            f"acc={test_acc:.2f}%  f1={test_f1:.4f}  "
            f"auc={test_auc:.4f}  sens={test_sens:.4f}  spec={test_spec:.4f}"
        )

    if not all_metrics:
        print(f"  No folds completed for {model_type.upper()}.")
        return {}

    # ── Aggregate across folds
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

    print(
        f"\n  [{model_type.upper()}] Mean over {len(all_metrics)} fold(s):\n"
        f"    Acc         : {mean_acc:.2f} ± {std_acc:.2f} %\n"
        f"    F1          : {mean_f1:.4f} ± {std_f1:.4f}\n"
        f"    AUC         : {mean_auc:.4f} ± {std_auc:.4f}\n"
        f"    Sensitivity : {mean_sens:.4f} ± {std_sens:.4f}\n"
        f"    Specificity : {mean_spec:.4f} ± {std_spec:.4f}"
    )

    # ── Write per-model results file
    results_path = os.path.join(model_dir, "test_results.txt")
    with open(results_path, "w") as f:
        f.write(f"model\t{model_type}\n")
        f.write(f"data\trest_only\n")
        f.write(f"k_fold\t{args.k_fold}\n")
        f.write(f"folds_run\t{len(all_metrics)}\n")
        f.write(f"num_epochs\t{args.num_epochs}\n")
        f.write(f"hidden_dim\t{args.hidden_dim}\n")
        f.write(f"top_k_edges\t{args.top_k_edges}\n")
        f.write(f"window_num\t{args.window_num}\n")
        f.write(f"window_size\t{args.window_size}\n")
        f.write("\n")
        f.write(f"acc_mean\t{mean_acc:.4f}\n")
        f.write(f"acc_std\t{std_acc:.4f}\n")
        f.write(f"f1_mean\t{mean_f1:.4f}\n")
        f.write(f"f1_std\t{std_f1:.4f}\n")
        f.write(f"auc_mean\t{mean_auc:.4f}\n")
        f.write(f"auc_std\t{std_auc:.4f}\n")
        f.write(f"sensitivity_mean\t{mean_sens:.4f}\n")
        f.write(f"sensitivity_std\t{std_sens:.4f}\n")
        f.write(f"specificity_mean\t{mean_spec:.4f}\n")
        f.write(f"specificity_std\t{std_spec:.4f}\n")
        f.write("\n")
        for i, m in enumerate(all_metrics):
            f.write(
                f"fold_{i}\t"
                f"acc={m['acc']:.4f}\tf1={m['f1']:.4f}\t"
                f"auc={m['auc']:.4f}\tsensitivity={m['sensitivity']:.4f}\t"
                f"specificity={m['specificity']:.4f}\n"
            )
    print(f"  Results written → {results_path}")

    return {
        "model": model_type,
        "acc":  (mean_acc,  std_acc),
        "f1":   (mean_f1,   std_f1),
        "auc":  (mean_auc,  std_auc),
        "sens": (mean_sens, std_sens),
        "spec": (mean_spec, std_spec),
        "n_folds": len(all_metrics),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    os.makedirs(args.targetdir, exist_ok=True)

    # ── Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda requested but CUDA unavailable; running on CPU.")

    print(f"\nBiopoint REST-ONLY classification")
    print(f"  sourcedir  : {args.sourcedir}")
    print(f"  csv_path   : {args.csv_path}")
    print(f"  targetdir  : {args.targetdir}")
    print(f"  device     : {device}")
    print(f"  k_fold     : {args.k_fold}  (running {args.max_folds} fold(s))")
    print(f"  epochs     : {args.num_epochs}")
    print(f"  models     : {args.models}")

    all_results = []
    for model_type in args.models:
        torch.manual_seed(args.seed)   # reset seed per model for reproducibility
        np.random.seed(args.seed)
        random.seed(args.seed)
        result = run_model(model_type, args, device)
        if result:
            all_results.append(result)

    # ── Combined summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY  —  Biopoint rest-only  ({args.max_folds}/{args.k_fold} fold(s))")
    print(f"{'='*60}")
    header = f"{'Model':<8}  {'Acc (%)':>12}  {'F1':>12}  {'AUC':>12}  {'Sens':>12}  {'Spec':>12}"
    print(header)
    print("-" * len(header))
    summary_lines = [header, "-" * len(header)]
    for r in all_results:
        line = (
            f"{r['model'].upper():<8}  "
            f"{r['acc'][0]:6.2f}±{r['acc'][1]:5.2f}  "
            f"{r['f1'][0]:.4f}±{r['f1'][1]:.4f}  "
            f"{r['auc'][0]:.4f}±{r['auc'][1]:.4f}  "
            f"{r['sens'][0]:.4f}±{r['sens'][1]:.4f}  "
            f"{r['spec'][0]:.4f}±{r['spec'][1]:.4f}"
        )
        print(line)
        summary_lines.append(line)

    summary_path = os.path.join(args.targetdir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Biopoint REST-ONLY — {args.max_folds}/{args.k_fold} fold(s)\n")
        f.write(f"epochs={args.num_epochs}  hidden_dim={args.hidden_dim}  "
                f"top_k_edges={args.top_k_edges}  window_num={args.window_num}\n\n")
        f.write("\n".join(summary_lines) + "\n")
    print(f"\nSummary written → {summary_path}")


if __name__ == "__main__":
    main()
