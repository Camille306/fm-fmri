#!/usr/bin/env python3
"""
LSTM-GAN Training Script for Rest-to-Task fMRI Prediction

Same evaluation protocol as TimeGAN (subject-level dedup, test_results.txt, group_fc_pred.npy).
Architecture: RestEncoder LSTM -> (h0,c0); Generator LSTM (autoregressive) -> task; Discriminator LSTM -> real/fake.
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

_this_dir = Path(__file__).resolve().parent
_project_root = _this_dir.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_this_dir))

from dataset import HCPRestingFCDataset
from lstm_gan_model import LSTMGAN
from eval_viz import save_closest_subject_visualizations
from fc_metrics import compute_fc_topk_precision_recall_auc, topk_metric_keys

# Reuse dataset and metrics from TimeGAN script
from train_timegan import (
    FMRIWindowDataset,
    aggregate_subject_timeline,
    compute_functional_connectivity,
    compute_fc_similarity,
    compute_frequency_difference,
    _print_topk,
    _write_topk,
)


# ======================================================================
# LSTM-GAN training: G and D only (no embedder/recovery/supervisor)
# ======================================================================

def train_joint(model, loader, g_optimizer, d_optimizer, device, prediction_length, lambda_mse=10.0, label_smooth=0.1):
    model.train()
    total_g = 0.0
    total_d = 0.0
    n = 0
    criterion = nn.MSELoss()

    for batch in tqdm(loader, desc="Joint Train (LSTM-GAN)", leave=False):
        x_rest = batch["input"].to(device).float()
        x_task = batch["target"].to(device).float()

        # Discriminator
        d_optimizer.zero_grad()
        x_fake = model(x_rest, prediction_length=prediction_length)
        y_real = model.discriminator(x_task)
        y_fake = model.discriminator(x_fake.detach())
        smooth = label_smooth
        d_loss_real = -torch.mean(torch.log(y_real * (1 - smooth) + smooth * 0.1 + 1e-8))
        d_loss_fake = -torch.mean(torch.log(1 - y_fake + smooth * 0.1 + 1e-8))
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        nn.utils.clip_grad_norm_(model.discriminator.parameters(), 1.0)
        d_optimizer.step()

        # Generator: adversarial + MSE
        g_optimizer.zero_grad()
        x_fake = model(x_rest, prediction_length=prediction_length)
        y_fake = model.discriminator(x_fake)
        g_loss_adv = -torch.mean(torch.log(y_fake + 1e-8))
        g_loss_mse = criterion(x_fake, x_task)
        g_loss = g_loss_adv + lambda_mse * g_loss_mse
        g_loss.backward()
        nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
        nn.utils.clip_grad_norm_(model.generator.parameters(), 1.0)
        g_optimizer.step()

        total_g += float(g_loss.item())
        total_d += float(d_loss.item())
        n += 1

    return total_g / max(n, 1), total_d / max(n, 1)


def evaluate_subject_level_dedup_lstm_gan(model, loader, device, pred_len, return_for_viz=False):
    """Same protocol as TimeGAN: aggregate windows per subject, compute metrics."""
    model.eval()
    subj_pred_chunks = defaultdict(list)
    subj_tgt_chunks = defaultdict(list)
    subj_starts = defaultdict(list)
    subj_total_len = defaultdict(int)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval (LSTM-GAN subject-dedup)", leave=False):
            x_rest = batch["input"].to(device).float()
            sids = batch["subject_id"]
            starts = batch["task_start_idx"]
            x_pred = model(x_rest, prediction_length=pred_len)
            x_task = batch["target"]
            x_pred_np = x_pred.cpu().numpy()
            x_task_np = x_task.numpy()
            if not isinstance(sids, (list, tuple)):
                sids = list(sids)
            if not isinstance(starts, (list, tuple)):
                starts = list(starts)
            for b, sid in enumerate(sids):
                sid = str(sid)
                st = int(starts[b])
                subj_pred_chunks[sid].append(x_pred_np[b])
                subj_tgt_chunks[sid].append(x_task_np[b])
                subj_starts[sid].append(st)
                subj_total_len[sid] = max(subj_total_len[sid], st + pred_len)

    per_subj = {}
    for sid in sorted(subj_pred_chunks.keys()):
        total_len = subj_total_len[sid]
        pred_full = aggregate_subject_timeline(subj_pred_chunks[sid], subj_starts[sid], total_len)
        tgt_full = aggregate_subject_timeline(subj_tgt_chunks[sid], subj_starts[sid], total_len)
        mse = float(np.mean((pred_full - tgt_full) ** 2))
        mae = float(np.mean(np.abs(pred_full - tgt_full)))
        freq_diff = float(compute_frequency_difference(pred_full, tgt_full))
        fc_sim = float(compute_fc_similarity(pred_full, tgt_full))
        topk = compute_fc_topk_precision_recall_auc(pred_full, tgt_full)
        per_subj[sid] = {
            "mse": mse, "mae": mae, "freq_diff": freq_diff, "fc_similarity": fc_sim,
            **{k: v for k, v in topk.items() if k != "k_percentiles" and isinstance(v, (int, float))},
        }

    metrics = {"num_subjects": len(per_subj)}
    keys = ["mse", "mae", "freq_diff", "fc_similarity"] + list(topk_metric_keys())
    if len(per_subj) == 0:
        for k in keys:
            metrics[k] = float("nan")
            metrics[k + "_std"] = float("nan")
        return (metrics, subj_pred_chunks, subj_tgt_chunks, subj_starts, subj_total_len, per_subj) if return_for_viz else metrics

    for k in keys:
        vals = np.array([per_subj[s].get(k, float("nan")) for s in per_subj], dtype=np.float64)
        use_nan = k == "fc_similarity"
        m = np.nanmean(vals) if use_nan else np.mean(vals)
        s = np.nanstd(vals) if use_nan else np.std(vals)
        metrics[k] = float(m) if not (np.isnan(m) or np.isinf(m)) else 0.0
        metrics[k + "_std"] = float(s) if not (np.isnan(s) or np.isinf(s)) else 0.0

    if return_for_viz:
        return metrics, subj_pred_chunks, subj_tgt_chunks, subj_starts, subj_total_len, per_subj
    return metrics


def compute_and_return_cfid_fc(model, test_loader, device, pred_len):
    """Compute cFID-FC (conditional Fréchet on FC). Returns float or None on failure."""
    try:
        re_eval_dir = _project_root / "re_eval"
        if not re_eval_dir.exists():
            return None
        if str(re_eval_dir) not in sys.path:
            sys.path.insert(0, str(re_eval_dir))
        from fc_utils import cfid_fc
        real_list, gen_list = [], []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Collect real/gen for cFID-FC", leave=False):
                x_rest = batch["input"].to(device).float()
                x_task = batch["target"].to(device).float()
                x_pred = model(x_rest, prediction_length=pred_len)
                if x_pred.dim() == 2:
                    x_pred = x_pred.unsqueeze(1)
                real_list.append(x_task.cpu().numpy())
                gen_list.append(x_pred.cpu().numpy())
        X_real = np.concatenate(real_list, axis=0)
        X_gen = np.concatenate(gen_list, axis=0)
        rng = np.random.default_rng(42)
        return cfid_fc(X_real, X_gen, eps=1e-6, max_fc_dim=500, rng=rng)
    except Exception as e:
        print(f"[cFID-FC] Skipped: {e}", flush=True)
        return None


def main():
    parser = argparse.ArgumentParser("Train LSTM-GAN with FM-TS-matched eval")
    parser.add_argument("--data_root", type=str, default="./data/hcp-resting-fc", help="Resting-state data root (default matches fm_fmri.py and other baselines).")
    parser.add_argument("--task_root", type=str, default="./data/hcp-task-ts", help="Task timeseries root (default matches fm_fmri.py and other baselines).")
    parser.add_argument("--task_name", type=str, default="emotion")
    parser.add_argument("--lookback_length", type=int, default=512)
    parser.add_argument("--prediction_length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--max_samples_per_subject", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lambda_mse", type=float, default=10.0, help="MSE weight for G (reconstruction)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_lstm_gan_fmts_eval")
    parser.add_argument("--eval_only", action="store_true", help="Load best checkpoint, run test eval only")
    args = parser.parse_args()

    if not Path(args.data_root).is_absolute():
        args.data_root = str(_project_root / args.data_root)
    if not Path(args.task_root).is_absolute():
        args.task_root = str(_project_root / args.task_root)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    ds = HCPRestingFCDataset(
        data_root=args.data_root,
        task_root=args.task_root,
        task_name=args.task_name,
    )
    if args.prediction_length is None:
        sid = ds.subject_ids[0]
        task = ds.load_task_subject(sid)
        if task.ndim == 1:
            task = task.reshape(-1, 1)
        args.prediction_length = int(task.shape[0])

    train_ds = FMRIWindowDataset(
        dataset=ds,
        lookback_length=args.lookback_length,
        prediction_length=args.prediction_length,
        stride=args.stride,
        normalize=args.normalize,
        split="train",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_samples_per_subject=args.max_samples_per_subject,
    )
    val_ds = FMRIWindowDataset(
        dataset=ds,
        lookback_length=args.lookback_length,
        prediction_length=args.prediction_length,
        stride=args.stride,
        normalize=args.normalize,
        split="val",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_samples_per_subject=args.max_samples_per_subject,
    )
    test_ds = FMRIWindowDataset(
        dataset=ds,
        lookback_length=args.lookback_length,
        prediction_length=args.prediction_length,
        stride=args.stride,
        normalize=args.normalize,
        split="test",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_samples_per_subject=args.max_samples_per_subject,
    )
    if args.normalize and train_ds.rest_means is not None:
        for d in [val_ds, test_ds]:
            d.rest_means, d.rest_stds = train_ds.rest_means, train_ds.rest_stds
            d.task_means, d.task_stds = train_ds.task_means, train_ds.task_stds

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    V = int(next(iter(train_loader))["input"].shape[-1])

    model = LSTMGAN(
        input_dim=V,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_prediction_length=args.prediction_length,
    ).to(args.device)

    g_optimizer = Adam(
        list(model.encoder.parameters()) + list(model.generator.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    d_optimizer = Adam(
        model.discriminator.parameters(),
        lr=args.lr * 0.5,
        weight_decay=args.weight_decay,
    )
    g_sched = CosineAnnealingLR(g_optimizer, T_max=args.epochs)
    d_sched = CosineAnnealingLR(d_optimizer, T_max=args.epochs)

    best_path = os.path.join(args.save_dir, "best_lstm_gan_fmts_eval.pth")
    if args.eval_only:
        if not os.path.isfile(best_path):
            raise FileNotFoundError(f"eval_only: checkpoint not found at {best_path}")
        ckpt = torch.load(best_path, map_location=args.device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_metrics, subj_pred_chunks, subj_tgt_chunks, subj_starts, subj_total_len, per_subj = evaluate_subject_level_dedup_lstm_gan(
            model, test_loader, args.device, args.prediction_length, return_for_viz=True
        )
        print("\n" + "=" * 70)
        print("LSTM-GAN TEST (eval_only, Subject-level, Deduplicated) [BEST CKPT]")
        print("=" * 70)
        print(f"MSE (mean ± std): {test_metrics['mse']:.6f} ± {test_metrics['mse_std']:.6f}")
        print(f"MAE (mean ± std): {test_metrics['mae']:.6f} ± {test_metrics['mae_std']:.6f}")
        print(f"PSD (mean ± std): {test_metrics['freq_diff']:.6f} ± {test_metrics['freq_diff_std']:.6f}")
        print(f"FC sim (mean ± std): {test_metrics['fc_similarity']:.6f} ± {test_metrics['fc_similarity_std']:.6f}")
        _print_topk(test_metrics)
        print(f"Num subjects: {test_metrics['num_subjects']}")
        print("=" * 70)
        save_closest_subject_visualizations(
            per_subj, subj_pred_chunks, subj_tgt_chunks, subj_starts, subj_total_len,
            aggregate_subject_timeline,
            out_dir=args.save_dir,
            model_name="LSTM-GAN",
            fs=0.72,
        )
        fc_list = []
        for sid in sorted(subj_pred_chunks.keys()):
            total_len = subj_total_len[sid]
            pred_full = aggregate_subject_timeline(subj_pred_chunks[sid], subj_starts[sid], total_len)
            if pred_full is not None and pred_full.shape[0] >= 2:
                fc_list.append(np.clip(compute_functional_connectivity(pred_full), -1.0, 1.0))
        if fc_list:
            np.save(os.path.join(args.save_dir, "group_fc_pred.npy"), np.nanmean(np.stack(fc_list), axis=0))
        cfid_fc_value = None
        print("Computing cFID-FC (conditional Fréchet on FC)...", flush=True)
        cfid_fc_value = compute_and_return_cfid_fc(model, test_loader, args.device, args.prediction_length)
        if cfid_fc_value is not None:
            print(f"cFID-FC: {cfid_fc_value:.6f}  (lower = better)", flush=True)
        with open(os.path.join(args.save_dir, "test_results.txt"), "w", encoding="utf-8") as f:
            f.write("LSTM-GAN TEST (Subject-level, Deduplicated) [BEST CKPT]\n")
            f.write("=" * 60 + "\n")
            f.write(f"MSE (mean ± std): {test_metrics['mse']:.6f} ± {test_metrics['mse_std']:.6f}\n")
            f.write(f"MAE (mean ± std): {test_metrics['mae']:.6f} ± {test_metrics['mae_std']:.6f}\n")
            f.write(f"PSD (absolute power spectrum difference, mean ± std): {test_metrics['freq_diff']:.6f} ± {test_metrics['freq_diff_std']:.6f}\n")
            f.write(f"FC sim (mean ± std): {test_metrics['fc_similarity']:.6f} ± {test_metrics['fc_similarity_std']:.6f}\n")
            _write_topk(f, test_metrics)
            f.write(f"Num subjects: {test_metrics['num_subjects']}\n")
            if cfid_fc_value is not None:
                f.write(f"cFID-FC (conditional Fréchet on FC): {cfid_fc_value:.6f}  (lower = better)\n")
            f.write("=" * 60 + "\n")
        return

    best_val = float("inf")
    history = []
    hist_path = os.path.join(args.save_dir, "history.csv")
    for ep in range(1, args.epochs + 1):
        g_loss, d_loss = train_joint(
            model, train_loader, g_optimizer, d_optimizer, args.device,
            args.prediction_length,
            lambda_mse=args.lambda_mse,
        )
        val_metrics = evaluate_subject_level_dedup_lstm_gan(
            model, val_loader, args.device, args.prediction_length
        )
        g_sched.step()
        d_sched.step()
        print(f"\nEpoch {ep}/{args.epochs}  g_loss={g_loss:.6f}  d_loss={d_loss:.6f}")
        print(f"  Val MSE: {val_metrics['mse']:.6f}  Val FC: {val_metrics['fc_similarity']:.6f}")
        history.append({
            "epoch": ep, "g_loss": g_loss, "d_loss": d_loss,
            "val_mse": val_metrics["mse"], "val_fc": val_metrics["fc_similarity"],
            "num_subjects": val_metrics["num_subjects"],
        })
        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "best_val_mse": best_val,
            }, best_path)
            print(f"  Saved best -> {best_path}")
        if history:
            with open(hist_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
                w.writeheader()
                w.writerows(history)

    ckpt = torch.load(best_path, map_location=args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics, subj_pred_chunks, subj_tgt_chunks, subj_starts, subj_total_len, per_subj = evaluate_subject_level_dedup_lstm_gan(
        model, test_loader, args.device, args.prediction_length, return_for_viz=True
    )
    print("\n" + "=" * 70)
    print("LSTM-GAN TEST (Subject-level, Deduplicated) [BEST CKPT]")
    print("=" * 70)
    print(f"MSE (mean ± std): {test_metrics['mse']:.6f} ± {test_metrics['mse_std']:.6f}")
    print(f"MAE (mean ± std): {test_metrics['mae']:.6f} ± {test_metrics['mae_std']:.6f}")
    print(f"PSD (mean ± std): {test_metrics['freq_diff']:.6f} ± {test_metrics['freq_diff_std']:.6f}")
    print(f"FC sim (mean ± std): {test_metrics['fc_similarity']:.6f} ± {test_metrics['fc_similarity_std']:.6f}")
    _print_topk(test_metrics)
    print(f"Num subjects: {test_metrics['num_subjects']}")
    print("=" * 70)
    save_closest_subject_visualizations(
        per_subj, subj_pred_chunks, subj_tgt_chunks, subj_starts, subj_total_len,
        aggregate_subject_timeline,
        out_dir=args.save_dir,
        model_name="LSTM-GAN",
        fs=0.72,
    )
    fc_list = []
    for sid in sorted(subj_pred_chunks.keys()):
        total_len = subj_total_len[sid]
        pred_full = aggregate_subject_timeline(subj_pred_chunks[sid], subj_starts[sid], total_len)
        if pred_full is not None and pred_full.shape[0] >= 2:
            fc_list.append(np.clip(compute_functional_connectivity(pred_full), -1.0, 1.0))
    if fc_list:
        np.save(os.path.join(args.save_dir, "group_fc_pred.npy"), np.nanmean(np.stack(fc_list), axis=0))
    cfid_fc_value = None
    print("Computing cFID-FC (conditional Fréchet on FC)...", flush=True)
    cfid_fc_value = compute_and_return_cfid_fc(model, test_loader, args.device, args.prediction_length)
    if cfid_fc_value is not None:
        print(f"cFID-FC: {cfid_fc_value:.6f}  (lower = better)", flush=True)
    with open(os.path.join(args.save_dir, "test_results.txt"), "w", encoding="utf-8") as f:
        f.write("LSTM-GAN TEST (Subject-level, Deduplicated) [BEST CKPT]\n")
        f.write("=" * 60 + "\n")
        f.write(f"MSE (mean ± std): {test_metrics['mse']:.6f} ± {test_metrics['mse_std']:.6f}\n")
        f.write(f"MAE (mean ± std): {test_metrics['mae']:.6f} ± {test_metrics['mae_std']:.6f}\n")
        f.write(f"PSD (absolute power spectrum difference, mean ± std): {test_metrics['freq_diff']:.6f} ± {test_metrics['freq_diff_std']:.6f}\n")
        f.write(f"FC sim (mean ± std): {test_metrics['fc_similarity']:.6f} ± {test_metrics['fc_similarity_std']:.6f}\n")
        _write_topk(f, test_metrics)
        f.write(f"Num subjects: {test_metrics['num_subjects']}\n")
        if cfid_fc_value is not None:
            f.write(f"cFID-FC (conditional Fréchet on FC): {cfid_fc_value:.6f}  (lower = better)\n")
        f.write("=" * 60 + "\n")
    print("Test results written to", os.path.join(args.save_dir, "test_results.txt"))


if __name__ == "__main__":
    main()
