#!/usr/bin/env python3
"""
TimeGAN Training Script for Rest-to-Task fMRI Prediction
WITH FM-TS-MATCHED EVALUATION (subject-level dedup timeline)

Key differences vs your earlier TimeGAN version:
- Dataset returns: input, target, subject_id, task_start_idx
- Evaluation matches FM-TS: aggregate overlapping predicted windows per subject, then compute metrics on full timeline
- Save best checkpoint by VAL MSE using that subject-level eval
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from scipy import signal
from scipy.stats import pearsonr


# ----------------------------------------------------------------------
# Imports (run from project root or baselines/; dataset from project root)
# ----------------------------------------------------------------------
_this_dir = Path(__file__).resolve().parent   # baselines/
_project_root = _this_dir.parent              # project root
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_this_dir))

from dataset import HCPRestingFCDataset
from timegan_model import TimeGAN
from eval_viz import save_closest_subject_visualizations


# ======================================================================
# Windowed dataset that RETURNS subject_id and task_start_idx
# (This fixes your KeyError and enables FM-TS-style evaluation.)
# ======================================================================

class FMRIWindowDataset(Dataset):
    def __init__(
        self,
        dataset: HCPRestingFCDataset,
        lookback_length: int = 512,
        prediction_length: int = 176,
        stride: int = 10,
        normalize: bool = True,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        max_samples_per_subject: int = None,
        norm_sample_size: int = 1000,
        norm_batch_size: int = 100,
    ):
        self.dataset = dataset
        self.lookback_length = lookback_length
        self.prediction_length = prediction_length
        self.stride = stride
        self.normalize = normalize
        self.split = split
        self.max_samples_per_subject = max_samples_per_subject

        self.window_metadata = []
        self.rest_means = None
        self.rest_stds = None
        self.task_means = None
        self.task_stds = None

        self._create_window_indices(train_ratio, val_ratio)

        if self.normalize and len(self.window_metadata) > 0:
            self._compute_normalization_stats(sample_size=norm_sample_size, batch_size=norm_batch_size)

    def _create_window_indices(self, train_ratio: float, val_ratio: float):
        all_subjects = self.dataset.subject_ids
        n = len(all_subjects)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        if self.split == "train":
            subject_ids = all_subjects[:train_end]
        elif self.split == "val":
            subject_ids = all_subjects[train_end:val_end]
        else:
            subject_ids = all_subjects[val_end:]

        for subject_id in subject_ids:
            try:
                rest_ts = self.dataset.load_subject(subject_id)
                if rest_ts.ndim == 1:
                    rest_ts = rest_ts.reshape(-1, 1)
                R, V = rest_ts.shape

                task_ts = self.dataset.load_task_subject(subject_id)
                if task_ts.ndim == 1:
                    task_ts = task_ts.reshape(-1, 1)
                T, V2 = task_ts.shape
                if V2 != V:
                    continue

                max_rest_idx = R - self.lookback_length
                max_task_idx = T - self.prediction_length
                max_windows = min(max_rest_idx, max_task_idx)
                if max_windows < 0:
                    continue

                wcount = 0
                for rest_start in range(0, max_windows + 1, self.stride):
                    task_start = rest_start  # aligned indexing (same as your baseline)
                    if task_start + self.prediction_length > T:
                        break

                    self.window_metadata.append({
                        "subject_id": str(subject_id),
                        "rest_start_idx": int(rest_start),
                        "task_start_idx": int(task_start),
                    })

                    wcount += 1
                    if self.max_samples_per_subject and wcount >= self.max_samples_per_subject:
                        break

            except Exception:
                continue

    def _compute_normalization_stats(self, sample_size: int = 1000, batch_size: int = 100):
        if len(self.window_metadata) == 0:
            return

        m = min(sample_size, len(self.window_metadata))
        idxs = np.random.choice(len(self.window_metadata), m, replace=False)

        rest_sum = None
        rest_sum_sq = None
        rest_cnt = 0

        task_sum = None
        task_sum_sq = None
        task_cnt = 0

        V = None

        for s in range(0, len(idxs), batch_size):
            batch_idxs = idxs[s:s+batch_size]
            rest_batch = []
            task_batch = []

            for ii in batch_idxs:
                meta = self.window_metadata[ii]
                sid = meta["subject_id"]

                rest = self.dataset.load_subject(sid)
                if rest.ndim == 1:
                    rest = rest.reshape(-1, 1)
                if V is None:
                    V = rest.shape[1]

                rs = meta["rest_start_idx"]
                re = rs + self.lookback_length
                rest_batch.append(rest[rs:re].astype(np.float32))

                task = self.dataset.load_task_subject(sid)
                if task.ndim == 1:
                    task = task.reshape(-1, 1)
                ts = meta["task_start_idx"]
                te = ts + self.prediction_length
                task_batch.append(task[ts:te].astype(np.float32))

            if rest_batch:
                r = np.stack(rest_batch).reshape(-1, V)
                if rest_sum is None:
                    rest_sum = r.sum(0); rest_sum_sq = (r**2).sum(0)
                else:
                    rest_sum += r.sum(0); rest_sum_sq += (r**2).sum(0)
                rest_cnt += r.shape[0]

            if task_batch:
                t = np.stack(task_batch).reshape(-1, V)
                if task_sum is None:
                    task_sum = t.sum(0); task_sum_sq = (t**2).sum(0)
                else:
                    task_sum += t.sum(0); task_sum_sq += (t**2).sum(0)
                task_cnt += t.shape[0]

        self.rest_means = rest_sum / max(rest_cnt, 1)
        rest_var = (rest_sum_sq / max(rest_cnt, 1)) - self.rest_means**2
        self.rest_stds = np.sqrt(np.maximum(rest_var, 0.0))
        self.rest_stds = np.where(self.rest_stds < 1e-8, 1.0, self.rest_stds)

        self.task_means = task_sum / max(task_cnt, 1)
        task_var = (task_sum_sq / max(task_cnt, 1)) - self.task_means**2
        self.task_stds = np.sqrt(np.maximum(task_var, 0.0))
        self.task_stds = np.where(self.task_stds < 1e-8, 1.0, self.task_stds)

    def __len__(self):
        return len(self.window_metadata)

    def __getitem__(self, idx):
        meta = self.window_metadata[idx]
        sid = meta["subject_id"]

        rest = self.dataset.load_subject(sid)
        if rest.ndim == 1:
            rest = rest.reshape(-1, 1)
        rs = meta["rest_start_idx"]
        re = rs + self.lookback_length
        x = rest[rs:re].astype(np.float32)

        task = self.dataset.load_task_subject(sid)
        if task.ndim == 1:
            task = task.reshape(-1, 1)
        ts = meta["task_start_idx"]
        te = ts + self.prediction_length
        y = task[ts:te].astype(np.float32)

        if self.normalize and self.rest_means is not None:
            x = (x - self.rest_means) / self.rest_stds
            y = (y - self.task_means) / self.task_stds

        return {
            "input": torch.from_numpy(x),          # (L,V)
            "target": torch.from_numpy(y),         # (T,V)
            "subject_id": sid,
            "task_start_idx": int(ts),
        }


# ======================================================================
# Metrics (same as FM-TS)
# ======================================================================

def compute_functional_connectivity(data: np.ndarray) -> np.ndarray:
    fc = np.corrcoef(data.T)
    return np.nan_to_num(fc, nan=0.0, posinf=1.0, neginf=-1.0)

def compute_fc_similarity(pred: np.ndarray, target: np.ndarray) -> float:
    try:
        fc_pred = compute_functional_connectivity(pred)
        fc_tgt  = compute_functional_connectivity(target)
        mask = np.triu(np.ones_like(fc_pred, dtype=bool), k=1)
        a = np.nan_to_num(fc_pred[mask].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        b = np.nan_to_num(fc_tgt[mask].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        if len(a) > 1 and np.std(a) > 1e-10 and np.std(b) > 1e-10:
            r, _ = pearsonr(a, b)
            out = float(r)
        else:
            out = 0.0
    except Exception:
        out = 0.0
    return 0.0 if (np.isnan(out) or np.isinf(out)) else out

def compute_frequency_difference(pred: np.ndarray, target: np.ndarray, fs: float = 0.72) -> float:
    # MAE of PSD
    V = pred.shape[1]
    diffs = []
    for v in range(V):
        try:
            _, p1 = signal.welch(pred[:, v], fs=fs, nperseg=min(64, pred.shape[0]))
            _, p2 = signal.welch(target[:, v], fs=fs, nperseg=min(64, target.shape[0]))
            m = min(len(p1), len(p2))
            diffs.append(np.mean(np.abs(p1[:m] - p2[:m])))
        except Exception:
            continue
    return float(np.mean(diffs)) if diffs else 0.0


# ======================================================================
# FM-TS-style dedup aggregation
# ======================================================================

def aggregate_subject_timeline(chunks, starts, total_len):
    """
    Deduplicate by averaging overlaps.
    chunks: list of (T,V)
    starts: list of start indices
    total_len: length of reconstructed series
    """
    V = chunks[0].shape[1]
    acc = np.zeros((total_len, V), dtype=np.float32)
    cnt = np.zeros((total_len, 1), dtype=np.float32)

    for x, s in zip(chunks, starts):
        T = x.shape[0]
        e = min(s + T, total_len)
        t_eff = e - s
        if t_eff <= 0:
            continue
        acc[s:e] += x[:t_eff]
        cnt[s:e] += 1.0

    cnt = np.maximum(cnt, 1.0)
    return acc / cnt


def evaluate_subject_level_dedup_timegan(model, loader, device, pred_len: int, return_for_viz: bool = False):
    """
    EXACT SAME evaluation idea as FM-TS:
      - collect predicted windows per subject (and targets)
      - dedup to a single subject timeline using task_start_idx
      - compute metrics per subject
      - average across subjects
    """
    model.eval()

    subj_pred_chunks = defaultdict(list)
    subj_tgt_chunks  = defaultdict(list)
    subj_starts      = defaultdict(list)
    subj_total_len   = defaultdict(int)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval (TimeGAN subject-dedup)", leave=False):
            x_rest = batch["input"].to(device).float()      # (B,L,V)
            x_task = batch["target"].to(device).float()     # (B,T,V)
            sids   = batch["subject_id"]
            starts = batch["task_start_idx"]

            # ---- TimeGAN inference ----
            x_pred = model(x_rest, prediction_length=pred_len)  # (B,T,V)

            x_pred_np = x_pred.detach().cpu().numpy()
            x_task_np = x_task.detach().cpu().numpy()

            # batch collation sometimes returns list[str], sometimes list/array
            if not isinstance(sids, (list, tuple)):
                sids = list(sids)
            if not isinstance(starts, (list, tuple)):
                starts = list(starts)

            for b, sid in enumerate(sids):
                sid = str(sid)
                st  = int(starts[b])

                subj_pred_chunks[sid].append(x_pred_np[b])
                subj_tgt_chunks[sid].append(x_task_np[b])
                subj_starts[sid].append(st)
                subj_total_len[sid] = max(subj_total_len[sid], st + pred_len)

    per_subj = {}
    for sid in sorted(subj_pred_chunks.keys()):
        total_len = subj_total_len[sid]
        pred_full = aggregate_subject_timeline(subj_pred_chunks[sid], subj_starts[sid], total_len)
        tgt_full  = aggregate_subject_timeline(subj_tgt_chunks[sid],  subj_starts[sid], total_len)

        mse = float(np.mean((pred_full - tgt_full) ** 2))
        mae = float(np.mean(np.abs(pred_full - tgt_full)))
        freq_diff = float(compute_frequency_difference(pred_full, tgt_full))
        fc_sim = float(compute_fc_similarity(pred_full, tgt_full))

        per_subj[sid] = {
            "mse": mse,
            "mae": mae,
            "freq_diff": freq_diff,
            "fc_similarity": fc_sim,
        }

    metrics = {"num_subjects": len(per_subj)}
    if len(per_subj) == 0:
        for k in ["mse", "mae", "freq_diff", "fc_similarity"]:
            metrics[k] = float("nan")
            metrics[k + "_std"] = float("nan")
        return (metrics, subj_pred_chunks, subj_tgt_chunks, subj_starts, subj_total_len, per_subj) if return_for_viz else metrics

    for k in ["mse", "mae", "freq_diff", "fc_similarity"]:
        vals = np.array([per_subj[s][k] for s in per_subj], dtype=np.float64)
        # FC can be nan for some subjects (e.g. constant pred); use nanmean/nanstd so aggregate is valid
        use_nan = (k == "fc_similarity")
        m = np.nanmean(vals) if use_nan else np.mean(vals)
        s = np.nanstd(vals) if use_nan else np.std(vals)
        metrics[k] = float(m) if not (np.isnan(m) or np.isinf(m)) else 0.0
        metrics[k + "_std"] = float(s) if not (np.isnan(s) or np.isinf(s)) else 0.0

    if return_for_viz:
        return metrics, subj_pred_chunks, subj_tgt_chunks, subj_starts, subj_total_len, per_subj
    return metrics


# ======================================================================
# TimeGAN training phases (standard)
# NOTE: this assumes your TimeGAN implements:
#   - model.embed(x)
#   - model.recover(h)
#   - model.generate_latent(x_rest, prediction_length)
#   - model.discriminate(h)
#   - model.supervise(h_seq)
#   - model(x_rest, prediction_length=...) -> x_pred
# ======================================================================

def train_embedder_recovery(model, loader, optimizer, device, prediction_length):
    model.train()
    model.embedder.train()
    model.recovery.train()
    model.generator.eval()
    model.discriminator.eval()
    model.supervisor.eval()

    criterion = nn.MSELoss()
    total = 0.0
    n = 0

    for batch in tqdm(loader, desc="Train Embedder/Recovery", leave=False):
        x_task = batch["target"].to(device).float()  # (B,T,V)

        optimizer.zero_grad()

        h_task = model.embed(x_task)                 # (B,T,H)
        x_rec  = model.recover(h_task)               # (B,T,V)
        loss   = criterion(x_rec, x_task)

        loss.backward()
        nn.utils.clip_grad_norm_(model.embedder.parameters(), 1.0)
        nn.utils.clip_grad_norm_(model.recovery.parameters(), 1.0)
        optimizer.step()

        total += float(loss.item())
        n += 1

    return total / max(n, 1)


def train_supervisor(model, loader, optimizer, device, prediction_length):
    model.train()
    model.embedder.train()     # must be train for cudnn backward if used
    model.supervisor.train()
    model.recovery.eval()
    model.generator.eval()
    model.discriminator.eval()

    criterion = nn.MSELoss()
    total = 0.0
    n = 0

    for batch in tqdm(loader, desc="Train Supervisor", leave=False):
        x_task = batch["target"].to(device).float()  # (B,T,V)

        optimizer.zero_grad()

        h_task = model.embed(x_task)                 # (B,T,H)
        h_hat  = model.supervise(h_task[:, :-1, :])  # (B,T-1,H)
        loss   = criterion(h_hat, h_task[:, 1:, :])

        loss.backward()
        optimizer.step()

        total += float(loss.item())
        n += 1

    return total / max(n, 1)


def train_joint(model, loader, g_optimizer, d_optimizer, device, prediction_length,
                lambda_embed=10.0, lambda_supervise=1.0, lambda_adv=0.1):
    model.train()
    model.embedder.train()
    model.recovery.train()
    model.supervisor.train()
    model.generator.train()
    model.discriminator.train()

    criterion = nn.MSELoss()
    total_g = 0.0
    total_d = 0.0
    n = 0

    for batch in tqdm(loader, desc="Joint Train", leave=False):
        x_rest = batch["input"].to(device).float()   # (B,L,V)
        x_task = batch["target"].to(device).float()  # (B,T,V)

        # ----------------- Discriminator -----------------
        d_optimizer.zero_grad()

        h_real = model.embed(x_task)                                     # (B,T,H)
        y_real = model.discriminate(h_real)                              # (B,T,1)

        h_fake = model.generate_latent(x_rest, prediction_length)        # (B,T,H)
        y_fake = model.discriminate(h_fake.detach())                     # (B,T,1)

        label_smooth = 0.1
        d_loss_real = -torch.mean(torch.log(y_real * (1 - label_smooth) + label_smooth * 0.1 + 1e-8))
        d_loss_fake = -torch.mean(torch.log(1 - y_fake + label_smooth * 0.1 + 1e-8))
        d_loss = d_loss_real + d_loss_fake

        d_loss.backward()
        nn.utils.clip_grad_norm_(model.discriminator.parameters(), 1.0)
        d_optimizer.step()

        # ----------------- Generator -----------------
        g_optimizer.zero_grad()

        h_fake = model.generate_latent(x_rest, prediction_length)        # (B,T,H)
        y_fake = model.discriminate(h_fake)                              # (B,T,1)
        g_loss_adv = -torch.mean(torch.log(y_fake + 1e-8))

        x_rec = model.recover(h_fake)                                    # (B,T,V)
        g_loss_embed = criterion(x_rec, x_task)

        h_sup = model.supervise(h_fake[:, :-1, :])                       # (B,T-1,H)
        g_loss_sup = criterion(h_sup, h_fake[:, 1:, :])

        g_loss = lambda_adv * g_loss_adv + lambda_embed * g_loss_embed + lambda_supervise * g_loss_sup

        g_loss.backward()
        nn.utils.clip_grad_norm_(model.generator.parameters(), 1.0)
        nn.utils.clip_grad_norm_(model.recovery.parameters(), 1.0)
        g_optimizer.step()

        total_g += float(g_loss.item())
        total_d += float(d_loss.item())
        n += 1

    return total_g / max(n, 1), total_d / max(n, 1)


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser("Train TimeGAN with FM-TS-matched eval")

    # Data
    parser.add_argument("--data_root", type=str, default="./data/hcp-resting-fc")
    parser.add_argument("--task_root", type=str, default="./data/hcp-task-ts")
    parser.add_argument("--task_name", type=str, default="emotion")
    parser.add_argument("--lookback_length", type=int, default=512)
    parser.add_argument("--prediction_length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--max_samples_per_subject", type=int, default=None)

    # Model
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--embedder_epochs", type=int, default=20)
    parser.add_argument("--supervisor_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--lambda_embed", type=float, default=10.0)
    parser.add_argument("--lambda_supervise", type=float, default=1.0)
    parser.add_argument("--lambda_adv", type=float, default=0.1)

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_timegan_fmts_eval")

    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Using device: {args.device}")

    # Dataset loader
    ds = HCPRestingFCDataset(
        data_root=args.data_root,
        task_root=args.task_root,
        task_name=args.task_name
    )
    print(f"Subjects: {len(ds)}")

    if args.prediction_length is None:
        sid = ds.subject_ids[0]
        task = ds.load_task_subject(sid)
        if task.ndim == 1:
            task = task.reshape(-1, 1)
        args.prediction_length = int(task.shape[0])
        print(f"Inferred prediction_length={args.prediction_length}")

    # Window datasets (with task_start_idx!)
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

    # Share normalization stats across splits
    if args.normalize and train_ds.rest_means is not None:
        for d in [val_ds, test_ds]:
            d.rest_means, d.rest_stds = train_ds.rest_means, train_ds.rest_stds
            d.task_means, d.task_stds = train_ds.task_means, train_ds.task_stds

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Infer V
    sample = next(iter(train_loader))
    V = int(sample["input"].shape[-1])

    # Model
    model = TimeGAN(
        input_dim=V,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_prediction_length=args.prediction_length,
    ).to(args.device)
    print(f"TimeGAN params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Optimizers
    embedder_optimizer = Adam(
        list(model.embedder.parameters()) + list(model.recovery.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    supervisor_optimizer = Adam(
        model.supervisor.parameters(),
        lr=args.lr, weight_decay=args.weight_decay
    )
    g_optimizer = Adam(
        list(model.generator.parameters()) + list(model.recovery.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    d_optimizer = Adam(
        model.discriminator.parameters(),
        lr=args.lr * 0.5, weight_decay=args.weight_decay
    )

    g_sched = CosineAnnealingLR(g_optimizer, T_max=args.epochs)
    d_sched = CosineAnnealingLR(d_optimizer, T_max=args.epochs)

    # ------------------------------------------------------------------
    # Phase 1: Embedder/Recovery
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Phase 1: Embedder/Recovery Pretrain")
    print("=" * 70)
    for ep in range(1, args.embedder_epochs + 1):
        loss = train_embedder_recovery(model, train_loader, embedder_optimizer, args.device, args.prediction_length)
        print(f"  [E/R] Epoch {ep:02d}/{args.embedder_epochs:02d}  loss={loss:.6f}")

    # ------------------------------------------------------------------
    # Phase 2: Supervisor
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Phase 2: Supervisor Pretrain")
    print("=" * 70)
    for ep in range(1, args.supervisor_epochs + 1):
        loss = train_supervisor(model, train_loader, supervisor_optimizer, args.device, args.prediction_length)
        print(f"  [SUP] Epoch {ep:02d}/{args.supervisor_epochs:02d}  loss={loss:.6f}")

    # ------------------------------------------------------------------
    # Phase 3: Joint training + subject-level validation (FM-TS style)
    # ------------------------------------------------------------------
    best_val = float("inf")
    best_path = os.path.join(args.save_dir, "best_timegan_fmts_eval.pth")
    hist_path = os.path.join(args.save_dir, "history.csv")
    history = []

    print("\n" + "=" * 70)
    print("Phase 3: Joint Training (G + D) with Subject-level Dedup Val")
    print("=" * 70)

    for ep in range(1, args.epochs + 1):
        g_loss, d_loss = train_joint(
            model, train_loader, g_optimizer, d_optimizer, args.device,
            args.prediction_length,
            lambda_embed=args.lambda_embed,
            lambda_supervise=args.lambda_supervise,
            lambda_adv=args.lambda_adv,
        )

        val_metrics = evaluate_subject_level_dedup_timegan(
            model, val_loader, args.device, args.prediction_length
        )

        g_sched.step()
        d_sched.step()

        print(f"\nEpoch {ep}/{args.epochs}")
        print(f"  Joint g_loss={g_loss:.6f}  d_loss={d_loss:.6f}")
        print(f"  Val (subject-dedup) MSE: {val_metrics['mse']:.6f} ± {val_metrics['mse_std']:.6f}")
        print(f"  Val (subject-dedup) MAE: {val_metrics['mae']:.6f} ± {val_metrics['mae_std']:.6f}")
        print(f"  Val (subject-dedup) PSD: {val_metrics['freq_diff']:.6f} ± {val_metrics['freq_diff_std']:.6f}")
        print(f"  Val (subject-dedup) FC : {val_metrics['fc_similarity']:.6f} ± {val_metrics['fc_similarity_std']:.6f}")
        print(f"  Subjects: {val_metrics['num_subjects']}")

        history.append({
            "epoch": ep,
            "g_loss": g_loss,
            "d_loss": d_loss,
            "val_mse": val_metrics["mse"],
            "val_mae": val_metrics["mae"],
            "val_psd": val_metrics["freq_diff"],
            "val_fc": val_metrics["fc_similarity"],
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
            print(f"  Saved best checkpoint -> {best_path}")

        # write history each epoch (safe on cluster)
        with open(hist_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            w.writeheader()
            w.writerows(history)

    # ------------------------------------------------------------------
    # Load best + test evaluation (subject-dedup)
    # ------------------------------------------------------------------
    ckpt = torch.load(best_path, map_location=args.device)
    model.load_state_dict(ckpt["model_state_dict"])

    print("\nEvaluating BEST model on test set (subject-level dedup)...")
    test_metrics, subj_pred_chunks, subj_tgt_chunks, subj_starts, subj_total_len, per_subj = evaluate_subject_level_dedup_timegan(
        model, test_loader, args.device, args.prediction_length, return_for_viz=True
    )

    print("\n" + "=" * 70)
    print("TimeGAN TEST (Subject-level, Deduplicated) [BEST CKPT]")
    print("=" * 70)
    print(f"MSE (mean ± std): {test_metrics['mse']:.6f} ± {test_metrics['mse_std']:.6f}")
    print(f"MAE (mean ± std): {test_metrics['mae']:.6f} ± {test_metrics['mae_std']:.6f}")
    print(f"PSD (abs power spectrum diff, mean ± std): {test_metrics['freq_diff']:.6f} ± {test_metrics['freq_diff_std']:.6f}")
    print(f"FC sim (mean ± std): {test_metrics['fc_similarity']:.6f} ± {test_metrics['fc_similarity_std']:.6f}")
    print(f"Num subjects: {test_metrics['num_subjects']}")
    print("=" * 70)

    save_closest_subject_visualizations(
        per_subj, subj_pred_chunks, subj_tgt_chunks, subj_starts, subj_total_len,
        aggregate_subject_timeline,
        out_dir=args.save_dir,
        model_name="TimeGAN",
        fs=0.72,
    )

    # Write best-fit test metrics to a single file for easy inspection
    test_results_path = os.path.join(args.save_dir, "test_results.txt")
    with open(test_results_path, "w", encoding="utf-8") as f:
        f.write("TimeGAN TEST (Subject-level, Deduplicated) [BEST CKPT]\n")
        f.write("=" * 60 + "\n")
        f.write(f"MSE (mean ± std): {test_metrics['mse']:.6f} ± {test_metrics['mse_std']:.6f}\n")
        f.write(f"MAE (mean ± std): {test_metrics['mae']:.6f} ± {test_metrics['mae_std']:.6f}\n")
        f.write(f"PSD (absolute power spectrum difference, mean ± std): {test_metrics['freq_diff']:.6f} ± {test_metrics['freq_diff_std']:.6f}\n")
        f.write(f"FC sim (mean ± std): {test_metrics['fc_similarity']:.6f} ± {test_metrics['fc_similarity_std']:.6f}\n")
        f.write(f"Num subjects: {test_metrics['num_subjects']}\n")
        f.write("=" * 60 + "\n")
    print(f"Test results written to {test_results_path}")


if __name__ == "__main__":
    main()
