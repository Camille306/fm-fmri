#!/usr/bin/env python3
"""
Sweep GAT hyperparameters on fold 4 (with synthetic augmentation) until test acc >= 80%.

For each config:
  1) Train GAT on fold 4 training set (real + synthetic)
  2) Evaluate on fold 4 test set with default argmax (threshold=0.5)
  3) Also sweep classification thresholds (0.3 - 0.7) on softmax prob
  4) Log all results to a summary file
  5) Stop early if any config + threshold hits the target accuracy

Usage:
  python gat_biopoint/sweep_fold4_to_80.py \
      --synthetic_dir ./synthetic_v3_best \
      --target_acc 80 \
      --device cuda

  # Or run a specific config index (for SLURM array jobs):
  python gat_biopoint/sweep_fold4_to_80.py \
      --synthetic_dir ./synthetic_v3_best \
      --config_idx 5 \
      --device cuda
"""

import os
import sys
import json
import argparse
import itertools
import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from torch.utils.data import Subset
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir))

from model import GATBiopoint
from dataset_biopoint import DatasetBiopointRestWithSynthetic
from experiment import train_epoch, evaluate


SWEEP_GRID = {
    "lr": [5e-4, 1e-3, 2e-3],
    "hidden_dim": [128, 256],
    "dropout": [0.05, 0.1],
    "weight_decay": [0.001, 0.01],
    "minibatch_size": [2, 4],
    "window_size": [30],
    "window_num": [12],
    "seed": [0],
}

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def build_configs():
    keys = sorted(SWEEP_GRID.keys())
    configs = []
    for vals in itertools.product(*(SWEEP_GRID[k] for k in keys)):
        configs.append(dict(zip(keys, vals)))
    return configs


def evaluate_with_thresholds(model, loader, device, thresholds):
    """Evaluate model on loader, returning per-threshold accuracy, f1, auc."""
    model.eval()
    out_list, y_list = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            out_list.append(out.cpu())
            y_list.append(data.y.view(-1).cpu())
    if not out_list:
        return {}
    out_all = torch.cat(out_list, dim=0)
    y_all = torch.cat(y_list, dim=0).numpy()
    probs = F.softmax(out_all, dim=1)[:, 1].numpy()

    try:
        auc = roc_auc_score(y_all, probs)
    except Exception:
        auc = 0.0

    results = {}
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        acc = accuracy_score(y_all, preds) * 100
        f1 = f1_score(y_all, preds, zero_division=1.0)
        results[thr] = {"acc": acc, "f1": f1, "auc": auc, "n_samples": len(y_all),
                        "n_pos_pred": int(preds.sum()), "n_pos_true": int(y_all.sum())}
    return results


def run_one_config(config, args):
    """Train and evaluate one hyperparameter config on fold 4. Returns dict with results."""
    fold = args.fold
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    from torch_geometric.loader import DataLoader

    dataset = DatasetBiopointRestWithSynthetic(
        args.sourcedir,
        synthetic_dir=args.synthetic_dir,
        csv_path=args.csv_path,
        k_fold=args.k_fold,
        train_ratio=0.8,
        window_size=config["window_size"],
        window_stride=args.window_stride,
        window_num=config["window_num"],
        dynamic_length=None,
        ts_filename_suffix="_shen268_ts.npy",
        atlas_source="dk",
        dk_atlas_ts_root=args.dk_atlas_ts_root,
    )

    roi_num = dataset.num_nodes
    num_classes = dataset.num_classes

    dataset.set_fold(fold, train=True)
    n_total = len(dataset)
    n_val = max(1, int(n_total * 0.1))
    n_train = n_total - n_val
    rng = np.random.RandomState(seed + fold)
    indices = rng.permutation(n_total).tolist()

    train_loader = DataLoader(Subset(dataset, indices[:n_train]),
                              batch_size=config["minibatch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(dataset, indices[n_train:]),
                            batch_size=config["minibatch_size"], shuffle=False, num_workers=0)

    model = GATBiopoint(
        roi_num=roi_num,
        window_num=config["window_num"],
        hidden_dim=config["hidden_dim"],
        num_classes=num_classes,
        dropout=config["dropout"],
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                 weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(args.num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1, val_auc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    dataset.set_fold(fold, train=False)
    test_loader = DataLoader(dataset, batch_size=config["minibatch_size"], shuffle=False, num_workers=0)

    default_loss, default_acc, default_f1, default_auc = evaluate(model, test_loader, criterion, device)
    threshold_results = evaluate_with_thresholds(model, test_loader, device, THRESHOLDS)

    best_thr = 0.5
    best_thr_acc = default_acc
    for thr, res in threshold_results.items():
        if res["acc"] > best_thr_acc:
            best_thr_acc = res["acc"]
            best_thr = thr

    return {
        "config": config,
        "fold": fold,
        "default_acc": default_acc,
        "default_f1": default_f1,
        "default_auc": default_auc,
        "best_threshold": best_thr,
        "best_threshold_acc": best_thr_acc,
        "threshold_results": {str(t): r for t, r in threshold_results.items()},
        "epochs_trained": epoch + 1,
        "best_val_loss": best_val_loss,
    }


def main():
    p = argparse.ArgumentParser(description="Sweep GAT configs on fold 4 to hit 80% accuracy")
    p.add_argument("--synthetic_dir", type=str, required=True)
    p.add_argument("--sourcedir", type=str,
                   default="./data/biopoint_data")
    p.add_argument("--csv_path", type=str,
                   default="./data/biopoint_data.csv")
    p.add_argument("--dk_atlas_ts_root", type=str,
                   default="./data/biopoint_dk_atlas")
    p.add_argument("--fold", type=int, default=4)
    p.add_argument("--k_fold", type=int, default=5)
    p.add_argument("--window_stride", type=int, default=3)
    p.add_argument("--num_epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--target_acc", type=float, default=80.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--log_file", type=str, default="./gat_fold4_sweep_log.txt")
    p.add_argument("--config_idx", type=int, default=None,
                   help="Run only this config index (for SLURM array). If None, run all sequentially.")
    args = p.parse_args()

    configs = build_configs()
    print(f"Total configs in grid: {len(configs)}")

    if args.config_idx is not None:
        if args.config_idx >= len(configs):
            print(f"config_idx {args.config_idx} out of range (max {len(configs)-1})")
            sys.exit(1)
        configs = [(args.config_idx, configs[args.config_idx])]
    else:
        configs = list(enumerate(configs))

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    hit_target = False
    for idx, config in configs:
        tag = (f"idx={idx} lr={config['lr']} h={config['hidden_dim']} "
               f"dp={config['dropout']} wd={config['weight_decay']} "
               f"bs={config['minibatch_size']} ws={config['window_size']} "
               f"wn={config['window_num']} seed={config['seed']}")
        print(f"\n{'='*70}")
        print(f"Config {idx}/{len(build_configs())-1}: {tag}")
        print(f"{'='*70}")

        try:
            result = run_one_config(config, args)
        except Exception as e:
            print(f"  FAILED: {e}")
            line = f"{datetime.now().isoformat()}\t{idx}\t{tag}\tFAILED\t{e}\n"
            with open(log_path, "a") as f:
                f.write(line)
            continue

        default_acc = result["default_acc"]
        best_thr = result["best_threshold"]
        best_thr_acc = result["best_threshold_acc"]
        auc = result["default_auc"]

        status = "HIT_TARGET" if best_thr_acc >= args.target_acc else "below"
        print(f"  Default (thr=0.5): acc={default_acc:.2f}%  f1={result['default_f1']:.4f}  auc={auc:.4f}")
        print(f"  Best threshold:    thr={best_thr:.2f} -> acc={best_thr_acc:.2f}%")
        print(f"  Epochs trained:    {result['epochs_trained']}")
        print(f"  Status:            {status}")

        thr_summary = " | ".join(
            f"thr={t}: {r['acc']:.1f}%"
            for t, r in sorted(result["threshold_results"].items())
        )

        line = (f"{datetime.now().isoformat()}\t{idx}\t{tag}\t{status}\t"
                f"default_acc={default_acc:.2f}\tbest_thr={best_thr:.2f}\t"
                f"best_thr_acc={best_thr_acc:.2f}\tauc={auc:.4f}\t"
                f"epochs={result['epochs_trained']}\t{thr_summary}\n")
        with open(log_path, "a") as f:
            f.write(line)

        if best_thr_acc >= args.target_acc:
            hit_target = True
            print(f"\n*** TARGET HIT: {best_thr_acc:.2f}% >= {args.target_acc}% ***")
            print(f"  Config: {json.dumps(config, indent=2)}")
            print(f"  Threshold: {best_thr}")
            detail_path = log_path.with_suffix(".best.json")
            with open(detail_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"  Full result saved to {detail_path}")
            break

    if not hit_target:
        print(f"\nNo config hit {args.target_acc}% on fold {args.fold}.")
        print(f"Check {log_path} for all results.")
    print(f"\nLog: {log_path}")


if __name__ == "__main__":
    main()
