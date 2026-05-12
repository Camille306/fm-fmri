"""
FM-TS (Conditional Flow Matching) Baseline for Rest-to-Task fMRI Prediction
Subject-level metrics with deduplicated subject timeline.

Core idea (Flow Matching):
- Sample x0 ~ N(0, I)  (same shape as task sequence)
- Let x1 = task target
- Sample t ~ Uniform(0,1)
- Define x_t = (1-t)*x0 + t*x1
- Target velocity v* = (x1 - x0)  (constant for linear interpolation)
- Train v_theta(t, x_t | rest) to match v* with MSE.

Inference:
- Start from x0 ~ N(0, I)
- Integrate ODE: dx/dt = v_theta(t, x(t) | rest) from t=0 -> 1 (Euler)
- Result x(1) is predicted task sequence.

Evaluation:
- Aggregate predictions per subject into a single timeline using task_start_idx
- Average overlaps (deduplicate)
- Compute metrics per subject and then average across subjects
"""

import os
import sys
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from scipy import signal
from scipy.stats import pearsonr

# -----------------------
# Imports (same pattern as yours)
# -----------------------
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from dataset import HCPRestingFCDataset

# If you already have FMRIWindowDataset somewhere, import it.
# IMPORTANT: We need task_start_idx (and task length) returned for dedup evaluation.
# If your existing FMRIWindowDataset doesn't return task_start_idx, use the one below (drop-in).
# from train import FMRIWindowDataset


# ======================================================================
# Windowed dataset (drop-in) that returns task_start_idx
# ======================================================================

class FMRIWindowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: HCPRestingFCDataset,
        lookback_length: int = 512,
        prediction_length: int = 166,
        stride: int = 10,
        normalize: bool = True,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        max_samples_per_subject: int = None,
        use_task_target: bool = True,
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
        self.use_task_target = use_task_target and (dataset.task_root is not None)

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

                if not self.use_task_target:
                    continue

                task_ts = self.dataset.load_task_subject(subject_id)
                if task_ts.ndim == 1:
                    task_ts = task_ts.reshape(-1, 1)
                T, V2 = task_ts.shape
                if V2 != V:
                    continue

                max_rest_idx = R - self.lookback_length
                max_task_idx = T - self.prediction_length
                max_windows = min(max_rest_idx, max_task_idx)

                wcount = 0
                for rest_start in range(0, max_windows + 1, self.stride):
                    task_start = rest_start
                    if task_start + self.prediction_length > T:
                        break
                    self.window_metadata.append({
                        "subject_id": subject_id,
                        "rest_start_idx": rest_start,
                        "task_start_idx": task_start,
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
# Metrics (same definitions you’ve been using)
# ======================================================================

def compute_functional_connectivity(data: np.ndarray) -> np.ndarray:
    fc = np.corrcoef(data.T)
    return np.nan_to_num(fc, nan=0.0, posinf=1.0, neginf=-1.0)

def compute_fc_similarity(pred: np.ndarray, target: np.ndarray) -> float:
    fc_pred = compute_functional_connectivity(pred)
    fc_tgt  = compute_functional_connectivity(target)

    mask = np.triu(np.ones_like(fc_pred, dtype=bool), k=1)
    a = fc_pred[mask]
    b = fc_tgt[mask]
    if len(a) > 1 and np.std(a) > 1e-10 and np.std(b) > 1e-10:
        r, _ = pearsonr(a, b)
        return float(r) if not np.isnan(r) else 0.0
    return 0.0

def compute_frequency_difference(pred: np.ndarray, target: np.ndarray, fs: float = 0.72) -> float:
    # MAE of PSD
    V = pred.shape[1]
    diffs = []
    for v in range(V):
        try:
            f1, p1 = signal.welch(pred[:, v], fs=fs, nperseg=min(64, pred.shape[0]))
            f2, p2 = signal.welch(target[:, v], fs=fs, nperseg=min(64, target.shape[0]))
            m = min(len(p1), len(p2))
            diffs.append(np.mean(np.abs(p1[:m] - p2[:m])))
        except Exception:
            continue
    return float(np.mean(diffs)) if diffs else 0.0


# ======================================================================
# FM-TS model: Rest encoder + conditional velocity network
# ======================================================================

class RestEncoder(nn.Module):
    """Encode rest sequence (B,L,V) -> context (B,C)"""
    def __init__(self, v_dim: int, hidden: int = 256, layers: int = 2, out_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(v_dim, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x):
        h, _ = self.lstm(x)
        last = h[:, -1, :]
        return self.proj(last)

class TimeEmbedding(nn.Module):
    """Scalar t -> (B, D)"""
    def __init__(self, dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
        )

    def forward(self, t: torch.Tensor):
        # t: (B,)
        return self.mlp(t.view(-1, 1))

class VelocityNet(nn.Module):
    """
    v_theta(t, x_t | rest):
    x_t: (B,T,V)
    rest_ctx: (B,C) -> broadcast to (B,T,C)
    t_emb: (B,D) -> broadcast to (B,T,D)
    output: (B,T,V)
    """
    def __init__(self, v_dim: int, ctx_dim: int = 256, t_dim: int = 128, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.in_dim = v_dim + ctx_dim + t_dim
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, v_dim),
        )

    def forward(self, x_t, rest_ctx, t_emb):
        B, T, V = x_t.shape
        ctx = rest_ctx.unsqueeze(1).expand(B, T, rest_ctx.shape[-1])
        te  = t_emb.unsqueeze(1).expand(B, T, t_emb.shape[-1])
        inp = torch.cat([x_t, ctx, te], dim=-1)
        return self.net(inp)

class FMTS(nn.Module):
    def __init__(self, v_dim: int, rest_hidden: int = 256, ctx_dim: int = 256, t_dim: int = 128):
        super().__init__()
        self.rest_enc = RestEncoder(v_dim=v_dim, hidden=rest_hidden, out_dim=ctx_dim)
        self.t_emb = TimeEmbedding(dim=t_dim)
        self.vnet = VelocityNet(v_dim=v_dim, ctx_dim=ctx_dim, t_dim=t_dim)

    def velocity(self, t, x_t, x_rest):
        ctx = self.rest_enc(x_rest)
        te  = self.t_emb(t)
        return self.vnet(x_t, ctx, te)

    def sample(self, x_rest, T_pred: int, steps: int = 50):
        """
        Euler integrate from t=0 -> 1
        x_rest: (B,L,V)
        returns x1_hat: (B,T,V)
        """
        B, L, V = x_rest.shape
        x = torch.randn(B, T_pred, V, device=x_rest.device)
        dt = 1.0 / steps
        for k in range(steps):
            t = torch.full((B,), k * dt, device=x_rest.device)
            v = self.velocity(t, x, x_rest)
            x = x + dt * v
        return x


# ======================================================================
# Training / Evaluation
# ======================================================================

def flow_matching_loss(model: FMTS, x_rest, x_task):
    """
    x_rest: (B,L,V), x_task: (B,T,V)
    """
    B, T, V = x_task.shape
    x0 = torch.randn_like(x_task)  # noise
    x1 = x_task                    # target
    t = torch.rand(B, device=x_task.device)
    t_view = t.view(B, 1, 1)

    x_t = (1.0 - t_view) * x0 + t_view * x1
    v_star = (x1 - x0)

    v_pred = model.velocity(t, x_t, x_rest)
    return F.mse_loss(v_pred, v_star)

def train_epoch(model, loader, opt, device, max_grad_norm=1.0):
    model.train()
    total = 0.0
    n = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        x_rest = batch["input"].to(device).float()    # (B,L,V)
        x_task = batch["target"].to(device).float()   # (B,T,V)

        opt.zero_grad()
        loss = flow_matching_loss(model, x_rest, x_task)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()

        total += float(loss.item())
        n += 1
    return total / max(n, 1)

def aggregate_subject_timeline(chunks, starts, total_len):
    """
    Deduplicate by averaging overlaps.
    chunks: list of (T,V)
    starts: list of start indices
    total_len: length of full task series for this subject (in TRs)
    returns (total_len, V)
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

def evaluate_subject_level_dedup(model, loader, device, pred_len, ode_steps=50):
    """
    Subject-level evaluation with deduplicated subject timeline reconstruction.
    """
    model.eval()

    subj_pred_chunks = defaultdict(list)
    subj_tgt_chunks  = defaultdict(list)
    subj_starts      = defaultdict(list)

    # We also need true task total length per subject to reconstruct full series.
    # We'll infer it by taking max(task_start_idx + pred_len) seen for that subject.
    subj_total_len = defaultdict(int)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            x_rest = batch["input"].to(device).float()
            x_task = batch["target"].to(device).float()
            sids = batch["subject_id"]
            starts = batch["task_start_idx"]

            if not isinstance(sids, (list, tuple)):
                sids = list(sids)
            if not isinstance(starts, (list, tuple)):
                starts = list(starts)

            # Predict using only rest (conditional generation)
            x_pred = model.sample(x_rest, T_pred=pred_len, steps=ode_steps)  # (B,T,V)

            x_pred_np = x_pred.cpu().numpy()
            x_task_np = x_task.cpu().numpy()

            for b, sid in enumerate(sids):
                sid = str(sid)
                st = int(starts[b])

                subj_pred_chunks[sid].append(x_pred_np[b])  # (T,V)
                subj_tgt_chunks[sid].append(x_task_np[b])   # (T,V)
                subj_starts[sid].append(st)
                subj_total_len[sid] = max(subj_total_len[sid], st + pred_len)

    # Per-subject metrics
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
            "n_timepoints": int(total_len),
        }

    # Average across subjects
    keys = ["mse", "mae", "freq_diff", "fc_similarity"]
    metrics = {"num_subjects": len(per_subj)}
    if len(per_subj) == 0:
        for k in keys:
            metrics[k] = float("nan")
            metrics[k + "_std"] = float("nan")
        return metrics

    for k in keys:
        vals = np.array([per_subj[s][k] for s in per_subj], dtype=np.float64)
        metrics[k] = float(np.mean(vals))
        metrics[k + "_std"] = float(np.std(vals))

    return metrics

# =========================
# FM-TS TEST VISUALIZATIONS
#   (1) GT FC + Pred FC for closest subject
#   (2) Spectrum difference (closest subject)
# =========================

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ---- reuse your existing helpers ----
# compute_functional_connectivity(data) -> (V,V)
# compute_fc_similarity(pred, target) -> float
# compute_frequency_difference(pred, target) -> float  (optional)

def plot_fc_gt_vs_pred(pred_full: np.ndarray, tgt_full: np.ndarray, save_path: str, title_prefix: str = ""):
    """
    pred_full, tgt_full: (T, V)
    Saves a 1x2 figure: GT FC and Pred FC (same color scale).
    """
    fc_pred = compute_functional_connectivity(pred_full)
    fc_tgt  = compute_functional_connectivity(tgt_full)
    fc_sim  = compute_fc_similarity(pred_full, tgt_full)

    vmin = float(min(fc_pred.min(), fc_tgt.min()))
    vmax = float(max(fc_pred.max(), fc_tgt.max()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(fc_tgt, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
    axes[0].set_title("Ground Truth FC")
    axes[0].set_xlabel("ROI")
    axes[0].set_ylabel("ROI")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(fc_pred, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
    axes[1].set_title("Generated FC")
    axes[1].set_xlabel("ROI")
    axes[1].set_ylabel("ROI")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(f"{title_prefix}FC comparison | FC sim = {fc_sim:.4f}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_psd_spectrum_difference(pred_full: np.ndarray, tgt_full: np.ndarray, save_path: str, fs: float = 0.72):
    """
    pred_full, tgt_full: (T, V)
    Saves a 2x1 figure:
      - avg PSD pred vs gt
      - avg PSD diff (pred - gt)
    """
    T, V = pred_full.shape
    psd_pred_list, psd_tgt_list = [], []
    freqs = None

    for v in range(V):
        x = pred_full[:, v]
        y = tgt_full[:, v]
        if len(x) < 8:
            continue
        try:
            f1, p1 = signal.welch(x, fs=fs, nperseg=min(64, len(x)))
            f2, p2 = signal.welch(y, fs=fs, nperseg=min(64, len(y)))
            m = min(len(p1), len(p2))
            f1, p1 = f1[:m], p1[:m]
            f2, p2 = f2[:m], p2[:m]
            freqs = f1
            psd_pred_list.append(p1)
            psd_tgt_list.append(p2)
        except Exception:
            continue

    if not psd_pred_list:
        print("[WARN] PSD plotting skipped: no valid ROIs for Welch.")
        return

    psd_pred_avg = np.mean(np.stack(psd_pred_list, axis=0), axis=0)
    psd_tgt_avg  = np.mean(np.stack(psd_tgt_list, axis=0), axis=0)
    psd_diff     = psd_pred_avg - psd_tgt_avg

    psd_mse = float(np.mean((psd_pred_avg - psd_tgt_avg) ** 2))
    psd_mae = float(np.mean(np.abs(psd_diff)))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(freqs, psd_tgt_avg, linewidth=2, label="GT")
    axes[0].plot(freqs, psd_pred_avg, linewidth=2, label="Generated")
    axes[0].set_title("Average PSD (across ROIs)")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Power")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(freqs, psd_diff, linewidth=2)
    axes[1].axhline(0.0, linestyle="--", linewidth=1, alpha=0.7)
    axes[1].set_title(f"PSD Difference (Generated - GT) | PSD MSE={psd_mse:.6f}, PSD MAE={psd_mae:.6f}")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Δ Power")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_subject_level_dedup_with_best_subject(
    model, loader, device, pred_len, ode_steps=50,
    out_dir: str = None, fs: float = 0.72
):
    """
    Same as your evaluate_subject_level_dedup(), but:
      - tracks the 'closest subject' by MSE
      - optionally saves FC + PSD plots for that subject
    """
    from collections import defaultdict

    model.eval()
    subj_pred_chunks = defaultdict(list)
    subj_tgt_chunks  = defaultdict(list)
    subj_starts      = defaultdict(list)
    subj_total_len   = defaultdict(int)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval (subject-dedup + best subj)", leave=False):
            x_rest = batch["input"].to(device).float()
            x_task = batch["target"].to(device).float()
            sids   = batch["subject_id"]
            starts = batch["task_start_idx"]

            if not isinstance(sids, (list, tuple)):
                sids = list(sids)
            if not isinstance(starts, (list, tuple)):
                starts = list(starts)

            x_pred = model.sample(x_rest, T_pred=pred_len, steps=ode_steps)  # (B,T,V)
            x_pred_np = x_pred.detach().cpu().numpy()
            x_task_np = x_task.detach().cpu().numpy()

            for b, sid in enumerate(sids):
                sid = str(sid)
                st  = int(starts[b])

                subj_pred_chunks[sid].append(x_pred_np[b])
                subj_tgt_chunks[sid].append(x_task_np[b])
                subj_starts[sid].append(st)
                subj_total_len[sid] = max(subj_total_len[sid], st + pred_len)

    # Per-subject metrics + find best
    per_subj = {}
    best_sid = None
    best_mse = float("inf")
    best_pair = None  # (pred_full, tgt_full)

    for sid in sorted(subj_pred_chunks.keys()):
        total_len = subj_total_len[sid]
        pred_full = aggregate_subject_timeline(subj_pred_chunks[sid], subj_starts[sid], total_len)
        tgt_full  = aggregate_subject_timeline(subj_tgt_chunks[sid],  subj_starts[sid], total_len)

        mse = float(np.mean((pred_full - tgt_full) ** 2))
        mae = float(np.mean(np.abs(pred_full - tgt_full)))
        freq_diff = float(compute_frequency_difference(pred_full, tgt_full))
        fc_sim = float(compute_fc_similarity(pred_full, tgt_full))

        per_subj[sid] = {"mse": mse, "mae": mae, "freq_diff": freq_diff, "fc_similarity": fc_sim}

        if mse < best_mse:
            best_mse = mse
            best_sid = sid
            best_pair = (pred_full, tgt_full)

    # Aggregate metrics
    metrics = {"num_subjects": len(per_subj)}
    keys = ["mse", "mae", "freq_diff", "fc_similarity"]
    if len(per_subj) == 0:
        for k in keys:
            metrics[k] = float("nan")
            metrics[k + "_std"] = float("nan")
        return metrics

    for k in keys:
        vals = np.array([per_subj[s][k] for s in per_subj], dtype=np.float64)
        metrics[k] = float(np.mean(vals))
        metrics[k + "_std"] = float(np.std(vals))

    # Save plots for the closest subject
    if out_dir is not None and best_pair is not None:
        os.makedirs(out_dir, exist_ok=True)
        pred_full, tgt_full = best_pair

        fc_path = os.path.join(out_dir, f"best_subject_{best_sid}_fc_gt_vs_pred.png")
        psd_path = os.path.join(out_dir, f"best_subject_{best_sid}_psd_diff.png")

        plot_fc_gt_vs_pred(pred_full, tgt_full, fc_path, title_prefix=f"FM-TS best subject {best_sid} | ")
        plot_psd_spectrum_difference(pred_full, tgt_full, psd_path, fs=fs)

        print(f"[Saved] FC plot:  {fc_path}")
        print(f"[Saved] PSD plot: {psd_path}")
        print(f"[Best subject] sid={best_sid}  mse={best_mse:.6f}  fc={per_subj[best_sid]['fc_similarity']:.4f}")

    return metrics

# ======================================================================
# Main
# ======================================================================

def main():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--data_root", type=str, default="./data/hcp-resting-fc")
    p.add_argument("--task_root", type=str, default="./data/hcp-task-ts")
    p.add_argument("--task_name", type=str, default="emotion")
    p.add_argument("--lookback_length", type=int, default=512)
    p.add_argument("--prediction_length", type=int, default=None)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--normalize", action="store_true", default=True)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)

    # Model
    p.add_argument("--rest_hidden", type=int, default=256)
    p.add_argument("--ctx_dim", type=int, default=256)
    p.add_argument("--t_dim", type=int, default=128)

    # Training
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--ode_steps", type=int, default=50)

    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--save_dir", type=str, default="./checkpoints_fmts")

    args = p.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Device: {args.device}")

    ds = HCPRestingFCDataset(
        data_root=args.data_root,
        task_root=args.task_root,
        task_name=args.task_name
    )
    print(f"Subjects: {len(ds)}")

    # Infer prediction length from task
    if args.prediction_length is None:
        sid = ds.subject_ids[0]
        task = ds.load_task_subject(sid)
        if task.ndim == 1:
            task = task.reshape(-1, 1)
        args.prediction_length = int(task.shape[0])
        print(f"Inferred prediction_length={args.prediction_length}")

    train_ds = FMRIWindowDataset(
        dataset=ds,
        lookback_length=args.lookback_length,
        prediction_length=args.prediction_length,
        stride=args.stride,
        normalize=args.normalize,
        split="train",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        use_task_target=True,
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
        use_task_target=True,
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
        use_task_target=True,
    )

    # share normalization stats
    if args.normalize and train_ds.rest_means is not None:
        val_ds.rest_means, val_ds.rest_stds = train_ds.rest_means, train_ds.rest_stds
        test_ds.rest_means, test_ds.rest_stds = train_ds.rest_means, train_ds.rest_stds
        val_ds.task_means, val_ds.task_stds = train_ds.task_means, train_ds.task_stds
        test_ds.task_means, test_ds.task_stds = train_ds.task_means, train_ds.task_stds

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # infer V
    sample = next(iter(train_loader))
    V = int(sample["input"].shape[-1])
    print(f"V (ROIs) = {V}")

    model = FMTS(v_dim=V, rest_hidden=args.rest_hidden, ctx_dim=args.ctx_dim, t_dim=args.t_dim).to(args.device)
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float("inf")
    best_path = os.path.join(args.save_dir, "best_fmts.pth")

    for ep in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, opt, args.device, max_grad_norm=args.max_grad_norm)

        # subject-level dedup validation
        val_metrics = evaluate_subject_level_dedup(
            model, val_loader, args.device,
            pred_len=args.prediction_length,
            ode_steps=args.ode_steps
        )
        sched.step()

        print(f"\nEpoch {ep}/{args.epochs}")
        print(f"  Train FM loss: {tr_loss:.6f}")
        print(f"  Val (subject-avg) MSE: {val_metrics['mse']:.6f} ± {val_metrics['mse_std']:.6f}")
        print(f"  Val (subject-avg) MAE: {val_metrics['mae']:.6f} ± {val_metrics['mae_std']:.6f}")
        print(f"  Val (subject-avg) PSD: {val_metrics['freq_diff']:.6f} ± {val_metrics['freq_diff_std']:.6f}")
        print(f"  Val (subject-avg) FC : {val_metrics['fc_similarity']:.6f} ± {val_metrics['fc_similarity_std']:.6f}")
        print(f"  Subjects: {val_metrics['num_subjects']}")

        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)
            print(f"  Saved best -> {best_path}")

    # Test
    ckpt = torch.load(best_path, map_location=args.device)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate_subject_level_dedup_with_best_subject(
        model, test_loader, args.device,
    pred_len=args.prediction_length,
    ode_steps=args.ode_steps,
    out_dir=args.save_dir,       # <- enables saving plots
    fs=0.72                # <- TR-dependent (1/1.39 ≈ 0.72)
    )

    print("\n" + "=" * 70)
    print("FM-TS (Conditional Flow Matching) TEST (Subject-level, Dedup)")
    print("=" * 70)
    print(f"MSE (mean ± std): {test_metrics['mse']:.6f} ± {test_metrics['mse_std']:.6f}")
    print(f"MAE (mean ± std): {test_metrics['mae']:.6f} ± {test_metrics['mae_std']:.6f}")
    print(f"PSD MAE (mean ± std): {test_metrics['freq_diff']:.6f} ± {test_metrics['freq_diff_std']:.6f}")
    print(f"FC sim (mean ± std): {test_metrics['fc_similarity']:.6f} ± {test_metrics['fc_similarity_std']:.6f}")
    print(f"Num subjects: {test_metrics['num_subjects']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
