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

import math
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
# Imports: use dataset from same folder (fm-fmri)
# -----------------------
fm_fmri_dir = Path(__file__).parent
sys.path.insert(0, str(fm_fmri_dir))

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
        self.use_evs = getattr(dataset, "use_evs", False)

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

                if self.use_evs:
                    # Check EV availability but do NOT exclude subjects that lack EV files.
                    # Subjects without EVs will receive a zero EV tensor + zero mask at
                    # __getitem__ time (same fallback already in place), so they still train
                    # on the rest→task objective — they just contribute no EV conditioning
                    # signal.  This avoids dropping subjects and shrinking the training set.
                    _has_ev = False
                    try:
                        ev_ts = self.dataset.load_ev_subject(subject_id)
                        _has_ev = ev_ts.shape[0] > 0 and ev_ts.shape[1] >= 4
                    except (FileNotFoundError, ValueError):
                        _has_ev = False
                    # Always continue regardless of _has_ev — zero EV is handled in __getitem__

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

        out = {
            "input": torch.from_numpy(x),          # (L,V)
            "target": torch.from_numpy(y),         # (T,V)
            "subject_id": sid,
            "task_start_idx": int(ts),
        }
        # EV: full event table (N_events, 4) per subject, padded to max_events for batching
        MAX_EV_EVENTS = getattr(self, "max_ev_events", 64)
        if self.use_evs:
            try:
                ev_full = self.dataset.load_ev_subject(sid)  # (N_events, 4)
                N_ev, n_cols = ev_full.shape
                if N_ev > MAX_EV_EVENTS:
                    ev_full = ev_full[:MAX_EV_EVENTS]
                    N_ev = MAX_EV_EVENTS
                pad_len = MAX_EV_EVENTS - N_ev
                if pad_len > 0:
                    ev_full = np.concatenate([ev_full, np.zeros((pad_len, n_cols), dtype=np.float32)], axis=0)
                ev_mask = np.zeros(MAX_EV_EVENTS, dtype=np.float32)
                ev_mask[:N_ev] = 1.0
                out["ev"] = torch.from_numpy(ev_full)
                out["ev_mask"] = torch.from_numpy(ev_mask)
            except (FileNotFoundError, ValueError):
                out["ev"] = torch.zeros(MAX_EV_EVENTS, 4, dtype=torch.float32)
                out["ev_mask"] = torch.zeros(MAX_EV_EVENTS, dtype=torch.float32)
        else:
            out["ev"] = torch.zeros(MAX_EV_EVENTS, 4, dtype=torch.float32)
            out["ev_mask"] = torch.zeros(MAX_EV_EVENTS, dtype=torch.float32)
        return out


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


def _fc_upper_triangle_flat(fc: np.ndarray) -> np.ndarray:
    """Return upper-triangle (k=1) of FC as a 1D array. Shape (V,V) -> (N,) with N = V*(V-1)/2."""
    mask = np.triu(np.ones_like(fc, dtype=bool), k=1)
    return np.asarray(fc[mask], dtype=np.float64)


def compute_fc_topk_precision_recall_auc(
    pred: np.ndarray,
    target: np.ndarray,
    k_percentiles: tuple = (5, 10, 20, 50),
) -> dict:
    """
    For each k in k_percentiles (e.g. 5, 10, 20, 50):
    - Define ground-truth top k%% connectivities by |FC_target| (strongest edges).
    - Define predicted top k%% by |FC_pred|.
    - Precision@k = |intersection| / |pred_top_k| (of predicted top edges, how many are in GT top?).
    - Recall@k = |intersection| / |gt_top_k| (of GT top edges, how many did we recover?).
    - AUC@k = ROC-AUC when binary label = 1 if edge in GT top k%%, 0 else; score = |FC_pred|.

    Returns dict with keys like "precision_at_5", "recall_at_5", "auc_at_5", etc., and "k_percentiles".
    """
    fc_pred = compute_functional_connectivity(pred)
    fc_tgt = compute_functional_connectivity(target)
    pred_flat = _fc_upper_triangle_flat(fc_pred)
    gt_flat = _fc_upper_triangle_flat(fc_tgt)
    N = len(pred_flat)
    if N == 0:
        out = {}
        for k in k_percentiles:
            out[f"precision_at_{k}"] = out[f"recall_at_{k}"] = out[f"auc_at_{k}"] = float("nan")
        out["k_percentiles"] = k_percentiles
        return out

    # Use absolute value for "strength" of connectivity
    pred_abs = np.abs(pred_flat)
    gt_abs = np.abs(gt_flat)
    # Indices that would sort descending (strongest first)
    pred_order = np.argsort(-pred_abs)
    gt_order = np.argsort(-gt_abs)

    out = {}
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        roc_auc_score = None

    for k_pct in k_percentiles:
        n_top = max(1, int(round(N * k_pct / 100.0)))
        gt_top_idx = set(gt_order[:n_top])
        pred_top_idx = set(pred_order[:n_top])
        inter = len(gt_top_idx & pred_top_idx)
        prec = inter / len(pred_top_idx) if pred_top_idx else 0.0
        rec = inter / len(gt_top_idx) if gt_top_idx else 0.0
        out[f"precision_at_{k_pct}"] = float(prec)
        out[f"recall_at_{k_pct}"] = float(rec)

        # AUC@k: label = 1 if edge in GT top k%, else 0; score = pred strength (|FC_pred|)
        if roc_auc_score is not None:
            y_true = np.zeros(N, dtype=np.int32)
            y_true[list(gt_top_idx)] = 1
            y_score = pred_abs  # higher |pred| should rank positive higher
            if np.unique(y_true).size == 2:  # need both classes
                auc_k = roc_auc_score(y_true, y_score)
                out[f"auc_at_{k_pct}"] = float(auc_k)
            else:
                out[f"auc_at_{k_pct}"] = float("nan")
        else:
            out[f"auc_at_{k_pct}"] = float("nan")

    out["k_percentiles"] = k_percentiles
    return out


def compute_fc_topk_similarity(
    pred: np.ndarray,
    target: np.ndarray,
    k_percentiles: tuple = (5, 10, 20, 50),
) -> dict:
    """
    Compute FC similarity (Pearson correlation) after filtering to top k% connections.
    
    For each k in k_percentiles (e.g. 5, 10, 20, 50):
    - Compute FC matrices for both pred and target
    - Filter to top k% strongest connections (by absolute value) from ground truth
    - Compute Pearson correlation between filtered FC vectors
    - This measures how well the predicted FC matches ground truth for the strongest connections
    
    Args:
        pred: Predicted time series, shape (T, V) or (V,)
        target: Ground truth time series, shape (T, V) or (V,)
        k_percentiles: Tuple of k values (e.g., (5, 10, 20, 50) for top 5%, 10%, etc.)
    
    Returns:
        Dictionary with keys like "fc_similarity_at_5", "fc_similarity_at_10", etc., 
        and "k_percentiles".
    """
    fc_pred = compute_functional_connectivity(pred)
    fc_tgt = compute_functional_connectivity(target)
    
    # Get upper triangle (excluding diagonal)
    pred_flat = _fc_upper_triangle_flat(fc_pred)
    gt_flat = _fc_upper_triangle_flat(fc_tgt)
    N = len(pred_flat)
    
    if N == 0:
        out = {}
        for k in k_percentiles:
            out[f"fc_similarity_at_{k}"] = float("nan")
        out["k_percentiles"] = k_percentiles
        return out
    
    # Use absolute value to determine "strength" of connectivity
    gt_abs = np.abs(gt_flat)
    # Sort indices by ground truth strength (descending)
    gt_order = np.argsort(-gt_abs)
    
    out = {}
    
    for k_pct in k_percentiles:
        # Number of top connections to keep
        n_top = max(1, int(round(N * k_pct / 100.0)))
        
        # Get indices of top k% connections (based on ground truth)
        top_k_indices = gt_order[:n_top]
        
        # Filter both predicted and ground truth FC to top k% connections
        pred_filtered = pred_flat[top_k_indices]
        gt_filtered = gt_flat[top_k_indices]
        
        # Compute Pearson correlation between filtered FC vectors
        if len(pred_filtered) > 1 and np.std(pred_filtered) > 1e-10 and np.std(gt_filtered) > 1e-10:
            r, _ = pearsonr(pred_filtered, gt_filtered)
            similarity = float(r) if not np.isnan(r) else 0.0
        else:
            similarity = 0.0
        
        out[f"fc_similarity_at_{k_pct}"] = similarity
    
    out["k_percentiles"] = k_percentiles
    return out


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
# Learnable HRF kernel (basis = weighted sum of gamma-like functions)
# ======================================================================

def _gamma_hrf_basis(
    kernel_len: int,
    num_basis: int = 3,
    tr: float = 1.0,
) -> torch.Tensor:
    """
    Build HRF basis functions over time (in TRs). Returns (num_basis, kernel_len).
    Basis 0: canonical gamma (peak ~5s). Basis 1: temporal derivative. Basis 2: dispersion derivative (optional).
    Time grid: 0, tr, 2*tr, ..., (kernel_len-1)*tr.
    """
    t = torch.linspace(0, (kernel_len - 1) * tr, kernel_len, dtype=torch.float32)
    # Canonical double-gamma style: peak ~5s, undershoot ~15s (approximate)
    a1, a2 = 6.0, 16.0
    c1, c2 = 1.0, 1.0 / 6.0
    h = c1 * (t ** (a1 - 1)) * torch.exp(-t) - c2 * (t ** (a2 - 1)) * torch.exp(-t)
    h = h * (t >= 0).float()
    h = h / (h.sum() + 1e-8)

    basis_list = [h]
    if num_basis >= 2:
        # Temporal derivative (shifted difference)
        dt = t[1] - t[0] if len(t) > 1 else 1.0
        h_diff = torch.zeros_like(h)
        h_diff[1:] = (h[1:] - h[:-1]) / dt
        h_diff = h_diff / (h_diff.abs().sum() + 1e-8)
        basis_list.append(h_diff)
    if num_basis >= 3:
        # Second basis: later peak (dispersion-like)
        a1_late, a2_late = 8.0, 20.0
        h2 = c1 * (t ** (a1_late - 1)) * torch.exp(-t) - c2 * (t ** (a2_late - 1)) * torch.exp(-t)
        h2 = h2 * (t >= 0).float()
        h2 = h2 / (h2.sum() + 1e-8)
        basis_list.append(h2)
    for _ in range(num_basis - len(basis_list)):
        basis_list.append(h.clone())  # pad with canonical if num_basis > 3
    return torch.stack(basis_list[:num_basis], dim=0)


class LearnableHRFKernel(nn.Module):
    """
    Learnable 1D convolution kernel over time, parameterized as a linear combination
    of HRF basis functions. Can be shared across ROIs or per-ROI.
    Input (B, L, V) -> output (B, L, V) with same length (same padding).
    """

    def __init__(
        self,
        kernel_len: int = 20,
        num_basis: int = 3,
        v_dim: int = 1,
        per_roi: bool = False,
        tr: float = 1.0,
    ):
        super().__init__()
        self.kernel_len = kernel_len
        self.num_basis = num_basis
        self.v_dim = v_dim
        self.per_roi = per_roi
        self.register_buffer("_basis", _gamma_hrf_basis(kernel_len, num_basis, tr=tr).clone())  # (num_basis, kernel_len)
        if per_roi:
            c = torch.zeros(v_dim, num_basis)
            c[:, 0] = 1.0
            self.coef = nn.Parameter(c)
        else:
            self.coef = nn.Parameter(torch.tensor([1.0] + [0.0] * (num_basis - 1), dtype=torch.float32))
        self.pad = (kernel_len - 1) // 2

    def _get_kernel(self) -> torch.Tensor:
        # coef @ _basis: (num_basis,) @ (num_basis, L) -> (L,)  or  (V, num_basis) @ (num_basis, L) -> (V, L)
        if self.per_roi:
            k = (self.coef @ self._basis).clamp(min=1e-6)  # (V, kernel_len)
            k = k / k.sum(dim=1, keepdim=True)
            return k
        else:
            k = (self.coef @ self._basis).clamp(min=1e-6)  # (kernel_len,)
            k = k / k.sum()
            return k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, V). Convolve over dim=1 (time) with the learnable HRF kernel.
        """
        B, L, V = x.shape
        kernel = self._get_kernel()
        if self.per_roi:
            # (V, 1, kernel_len) for Conv1d with in_channels=V, out_channels=V, groups=V
            w = kernel.unsqueeze(1)
            x_t = x.transpose(1, 2)
            out = F.conv1d(x_t, w, padding=self.pad)
            return out.transpose(1, 2)[:, :L]
        else:
            w = kernel.unsqueeze(0).unsqueeze(0)
            x_t = x.transpose(1, 2)
            out = F.conv1d(x_t, w.expand(V, 1, self.kernel_len), groups=V, padding=self.pad)
            return out.transpose(1, 2)[:, :L]


# ======================================================================
# Event-conditioned HRF timecourse for attention K/V
# ======================================================================

def _build_boxcar_hard(
    onset_rel: torch.Tensor,
    duration: torch.Tensor,
    T: int,
) -> torch.Tensor:
    """
    Hard boxcar u(t) = 1 for onset_rel <= t < onset_rel + duration, else 0.
    onset_rel: (B, N, 1), duration: (B, N, 1). Returns (B, N, T).
    """
    t_grid = torch.arange(T, device=onset_rel.device, dtype=onset_rel.dtype).view(1, 1, -1)
    u = ((t_grid >= onset_rel) & (t_grid < onset_rel + duration)).float()
    return u


def _build_boxcar_smooth(
    onset_rel: torch.Tensor,
    duration: torch.Tensor,
    T: int,
    sigma: float = 0.5,
) -> torch.Tensor:
    """
    Smooth boxcar with sigmoids: u(t) rises at onset, falls at onset+duration.
    onset_rel: (B, N, 1), duration: (B, N, 1). Returns (B, N, T).
    """
    t_grid = torch.arange(T, device=onset_rel.device, dtype=onset_rel.dtype).view(1, 1, -1)
    rise = torch.sigmoid((t_grid - onset_rel) / max(sigma, 1e-5))
    fall = torch.sigmoid((onset_rel + duration - t_grid) / max(sigma, 1e-5))
    return rise * fall


class EventHRFTimecourse(nn.Module):
    """
    Build a boxcar u(t) from EV onset (col 0) and duration (col 1), convolve with
    per-event HRF basis to get timecourse. K is time-invariant: K_b,n = projK(z_b,n).
    V is timecourse-weighted event tokens. Timecourse normalized to max 1 per event.
    """

    def __init__(
        self,
        d_ev: int,
        num_basis: int = 3,
        kernel_len: int = 20,
        use_delay_width: bool = False,
        tr: float = 1.0,
        delay_max: float = 5.0,
        width_min: float = 0.5,
        width_max: float = 3.0,
        use_smooth_boxcar: bool = False,
        boxcar_sigma: float = 0.5,
    ):
        super().__init__()
        self.d_ev = d_ev
        self.num_basis = num_basis
        self.kernel_len = kernel_len
        self.use_delay_width = use_delay_width
        self.delay_max = delay_max
        self.width_min = width_min
        self.width_max = width_max
        self.use_smooth_boxcar = use_smooth_boxcar
        self.boxcar_sigma = boxcar_sigma
        self.register_buffer("_basis", _gamma_hrf_basis(kernel_len, num_basis, tr=tr).clone())
        self.hrf_weights_proj = nn.Linear(d_ev, num_basis)
        self.proj_k = nn.Linear(d_ev, d_ev)
        if use_delay_width:
            self.delay_width_proj = nn.Linear(d_ev, 2)

    def forward(
        self,
        event_tokens: torch.Tensor,
        ev: torch.Tensor,
        T: int,
        ev_mask: torch.Tensor,
        task_start_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        event_tokens: (B, N, d_ev), ev: (B, N, 4) [onset, duration, amplitude, condition], ev_mask: (B, N).
        If task_start_idx (B,) is given, EV col 0/1 are in global task TRs and we subtract start for this window.
        Returns:
          timecourse: (B, N, T) normalized to max 1 per event
          K: (B, N, d_ev) time-invariant keys (projK(z))
          V_tc: (B, T, N, d_ev) timecourse-weighted values
        """
        B, N, _ = event_tokens.shape
        device = event_tokens.device

        onset = ev[:, :, 0:1].float()
        duration = ev[:, :, 1:2].float().clamp(min=1e-3)
        if task_start_idx is not None:
            start = (
                task_start_idx
                if isinstance(task_start_idx, torch.Tensor)
                else torch.tensor(task_start_idx, device=device, dtype=onset.dtype)
            )
            if start.dim() == 1:
                start = start.view(B, 1, 1)
            onset = onset - start
        onset_rel = onset.clamp(min=0.0)

        if self.use_delay_width:
            dw = self.delay_width_proj(event_tokens)
            delay = (F.softplus(dw[:, :, 0:1]) + 0.0).clamp(max=self.delay_max)
            width = (F.softplus(dw[:, :, 1:2]) + 1.0).clamp(min=self.width_min, max=self.width_max)
            onset_rel = (onset_rel + delay).clamp(min=0.0)
            duration = duration * width

        if self.use_smooth_boxcar:
            boxcar = _build_boxcar_smooth(onset_rel, duration, T, sigma=self.boxcar_sigma)
        else:
            boxcar = _build_boxcar_hard(onset_rel, duration, T)

        hrf_weights = F.softmax(self.hrf_weights_proj(event_tokens), dim=-1)
        kernel = (hrf_weights @ self._basis).clamp(min=1e-6)
        kernel = kernel / kernel.sum(dim=-1, keepdim=True)

        pad = (self.kernel_len - 1) // 2
        boxcar_flat = boxcar.reshape(B * N, 1, T)
        kernel_flat = kernel.reshape(B * N, 1, self.kernel_len)
        timecourse = F.conv1d(boxcar_flat, kernel_flat, padding=pad, groups=B * N)
        timecourse = timecourse.reshape(B, N, T)
        timecourse = timecourse * ev_mask.unsqueeze(-1)

        tc_max = timecourse.amax(dim=2, keepdim=True).clamp(min=1e-8)
        timecourse = timecourse / tc_max

        K = self.proj_k(event_tokens)
        V_tc = (event_tokens.unsqueeze(2) * timecourse.unsqueeze(-1)).transpose(1, 2)
        return timecourse, K, V_tc


# ======================================================================
# FM-TS model: Rest encoder + conditional velocity network
# ======================================================================

class RestEncoder(nn.Module):
    """Encode rest sequence (B,L,V) -> context (B,C) with LSTM."""
    def __init__(self, v_dim: int, hidden: int = 256, layers: int = 2, out_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(v_dim, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x):
        h, _ = self.lstm(x)
        last = h[:, -1, :]
        return self.proj(last)


class RestTransformerEncoder(nn.Module):
    """
    Encode rest sequence (B, L, V) -> context (B, out_dim) with patch-based transformer.
    Uses a [CLS] token; output is the CLS embedding (rest fingerprint).
    """
    def __init__(
        self,
        v_dim: int,
        patch_len: int = 16,
        d_model: int = 256,
        num_layers: int = 2,
        nhead: int = 4,
        dim_feedforward: int = 512,
        out_dim: int = 256,
        dropout: float = 0.1,
        max_patches: int = 64,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.d_model = d_model
        patch_dim = patch_len * v_dim
        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches + 1, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.max_patches = max_patches
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, V = x.shape
        patch_len = self.patch_len
        remainder = L % patch_len
        if remainder != 0:
            pad_len = patch_len - remainder
            x = F.pad(x, (0, 0, 0, pad_len), value=0.0)
            L = L + pad_len
        num_patches = L // patch_len
        x = x.view(B, num_patches, patch_len * V)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : num_patches + 1]
        x = self.transformer(x)
        return self.proj(x[:, 0])


class PriorHead(nn.Module):
    """
    Head on rest context (CLS) that outputs parameters for a rest-conditioned x0 prior
    with subject-specific cross-ROI structure:
      - mean (B, V), std (B, V): per-ROI location and scale
      - U (B, V, K): low-rank factor loadings; sample z ~ N(0,1) (B, T, K), then
        corr = z @ U^T gives (B, T, V) correlated component
      x0 = mean + std*eps + corr  (eps ~ N(0,1) (B,T,V))
    So x0 has subject-specific cross-ROI structure before the ODE starts (helps FC).
    """
    def __init__(
        self,
        ctx_dim: int,
        v_dim: int,
        hidden: int = 128,
        prior_K: int = 8,
        min_std: float = 0.1,
        max_std: float = 2.0,
    ):
        super().__init__()
        self.v_dim = v_dim
        self.prior_K = prior_K
        self.min_std = min_std
        self.max_std = max_std
        self.shared = nn.Sequential(
            nn.Linear(ctx_dim, hidden),
            nn.GELU(),
        )
        self.mean_std_head = nn.Linear(hidden, 2 * v_dim)
        self.U_head = nn.Linear(hidden, v_dim * prior_K)

    def forward(self, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ctx: (B, ctx_dim). Returns mean (B, V), std (B, V), U (B, V, K).
        """
        h = self.shared(ctx)
        V = self.v_dim
        K = self.prior_K
        out_ms = self.mean_std_head(h)
        mean = out_ms[:, :V]
        log_std = out_ms[:, V:]
        std = (self.min_std + F.softplus(log_std)).clamp(max=self.max_std)
        U = self.U_head(h).view(-1, V, K)
        return mean, std, U

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


class EVEncoder(nn.Module):
    """
    Encode EV table: first 3 cols -> MLP; last col (condition) -> embedding; add -> event tokens.
    ev: (B, N_events, 4) -> (B, N_events, d_ev). Condition 0 = padding (masked out in cross-attn).
    """
    def __init__(self, num_conditions: int = 32, d_ev: int = 64, dropout: float = 0.1):
        super().__init__()
        self.d_ev = d_ev
        self.mlp_3 = nn.Sequential(
            nn.Linear(3, d_ev),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ev, d_ev),
        )
        self.condition_embed = nn.Embedding(num_conditions + 1, d_ev, padding_idx=0)

    def forward(self, ev: torch.Tensor) -> torch.Tensor:
        # ev: (B, N, 4) -> (B, N, d_ev)
        B, N, _ = ev.shape
        mlp_out = self.mlp_3(ev[:, :, :3])
        cond = ev[:, :, 3].long().clamp(0, self.condition_embed.num_embeddings - 1)
        emb = self.condition_embed(cond)
        return mlp_out + emb


class VelocityNet(nn.Module):
    """
    v_theta(t, x_t | rest [, event_tokens]):
    x_t: (B,T,V); rest_ctx: (B,C); t_emb: (B,D).
    If event_tokens and ev_mask are given:
      - If event_hrf_timecourse is set: build per-event HRF timecourses, use timecourse-weighted
        K/V so x_t queries attend to timecourse-conditioned event representations.
      - Else: standard cross-attend with Q from x_t, K/V = event_tokens.
    output: (B,T,V)
    """
    def __init__(
        self,
        v_dim: int,
        ctx_dim: int = 256,
        t_dim: int = 128,
        d_ev: int = 0,
        hidden: int = 512,
        dropout: float = 0.1,
        event_hrf_timecourse: "EventHRFTimecourse | None" = None,
    ):
        super().__init__()
        self.d_ev = d_ev
        self.event_hrf_timecourse = event_hrf_timecourse
        self.in_dim = v_dim + ctx_dim + t_dim + (d_ev if d_ev > 0 else 0)
        self.proj_q = nn.Linear(v_dim, d_ev, bias=False) if d_ev > 0 else None
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, v_dim),
        )

    def forward(
        self,
        x_t,
        rest_ctx,
        t_emb,
        event_tokens=None,
        ev_mask=None,
        ev=None,
        task_start_idx=None,
    ):
        B, T, V = x_t.shape
        ctx = rest_ctx.unsqueeze(1).expand(B, T, rest_ctx.shape[-1])
        te = t_emb.unsqueeze(1).expand(B, T, t_emb.shape[-1])

        if (
            event_tokens is not None
            and ev_mask is not None
            and self.d_ev > 0
            and ev_mask.any()
        ):
            Q = self.proj_q(x_t)
            scale = (self.d_ev ** -0.5)
            # has_events: (B,1,1) True for subjects with at least one valid event
            has_events = ev_mask.any(dim=-1, keepdim=True).unsqueeze(1)  # (B,1,1)
            # For subjects with no valid events, replace -inf row with 0 before softmax
            # to avoid NaN; event_ctx is zeroed out for those subjects afterwards.
            all_masked = ~has_events.expand_as(ev_mask.unsqueeze(1))   # (B,1,N)
            if self.event_hrf_timecourse is not None and ev is not None:
                _, K, V_tc = self.event_hrf_timecourse(
                    event_tokens, ev, T, ev_mask, task_start_idx=task_start_idx
                )
                attn = torch.bmm(Q, K.transpose(1, 2)) * scale
                attn = attn.masked_fill(ev_mask.unsqueeze(1) == 0, float("-inf"))
                attn = attn.masked_fill(all_masked, 0.0)
                attn = F.softmax(attn, dim=-1)
                event_ctx = (attn.unsqueeze(-1) * V_tc).sum(2)
            else:
                K = event_tokens
                V = event_tokens
                attn = torch.bmm(Q, K.transpose(1, 2)) * scale
                attn = attn.masked_fill(ev_mask.unsqueeze(1) == 0, float("-inf"))
                attn = attn.masked_fill(all_masked, 0.0)
                attn = F.softmax(attn, dim=-1)
                event_ctx = torch.bmm(attn, V)
            # Zero out event context for subjects that had no valid EV events
            event_ctx = event_ctx * has_events.float().expand_as(event_ctx)
            inp = torch.cat([x_t, ctx, te, event_ctx], dim=-1)
        else:
            if self.d_ev > 0:
                event_ctx = torch.zeros(B, T, self.d_ev, device=x_t.device, dtype=x_t.dtype)
                inp = torch.cat([x_t, ctx, te, event_ctx], dim=-1)
            else:
                inp = torch.cat([x_t, ctx, te], dim=-1)
        return self.net(inp)

class FMTS(nn.Module):
    def __init__(
        self,
        v_dim: int,
        rest_hidden: int = 256,
        ctx_dim: int = 256,
        t_dim: int = 128,
        rest_encoder: str = "transformer",
        rest_patch_len: int = 16,
        rest_num_layers: int = 2,
        rest_nhead: int = 4,
        rest_dim_feedforward: int = 512,
        use_evs: bool = False,
        num_conditions: int = 32,
        d_ev: int = 64,
        use_hrf_kernel: bool = False,
        hrf_kernel_len: int = 20,
        hrf_num_basis: int = 3,
        hrf_per_roi: bool = False,
        use_ev_hrf_timecourse: bool = False,
        ev_hrf_kernel_len: int = 20,
        ev_hrf_num_basis: int = 3,
        ev_hrf_use_delay_width: bool = True,
        ev_hrf_smooth_boxcar: bool = False,
        ev_hrf_boxcar_sigma: float = 0.5,
        prior_K: int = 8,
        use_prior_detach: bool = False,
    ):
        super().__init__()
        self.use_evs = use_evs
        self.use_prior_detach = use_prior_detach
        self.use_hrf_kernel = use_hrf_kernel
        if rest_encoder == "transformer":
            self.rest_enc = RestTransformerEncoder(
                v_dim=v_dim,
                patch_len=rest_patch_len,
                d_model=rest_hidden,
                num_layers=rest_num_layers,
                nhead=rest_nhead,
                dim_feedforward=rest_dim_feedforward,
                out_dim=ctx_dim,
            )
        else:
            self.rest_enc = RestEncoder(
                v_dim=v_dim,
                hidden=rest_hidden,
                layers=rest_num_layers,
                out_dim=ctx_dim,
            )
        self.t_emb = TimeEmbedding(dim=t_dim)
        self.ev_encoder = EVEncoder(num_conditions=num_conditions, d_ev=d_ev) if use_evs else None
        self.hrf_kernel = (
            LearnableHRFKernel(
                kernel_len=hrf_kernel_len,
                num_basis=hrf_num_basis,
                v_dim=v_dim,
                per_roi=hrf_per_roi,
            )
            if use_hrf_kernel
            else None
        )
        event_hrf_tc = None
        if use_evs and use_ev_hrf_timecourse:
            event_hrf_tc = EventHRFTimecourse(
                d_ev=d_ev,
                num_basis=ev_hrf_num_basis,
                kernel_len=ev_hrf_kernel_len,
                use_delay_width=ev_hrf_use_delay_width,
                use_smooth_boxcar=ev_hrf_smooth_boxcar,
                boxcar_sigma=ev_hrf_boxcar_sigma,
            )
        self.vnet = VelocityNet(
            v_dim=v_dim,
            ctx_dim=ctx_dim,
            t_dim=t_dim,
            d_ev=d_ev if use_evs else 0,
            event_hrf_timecourse=event_hrf_tc,
        )
        self.prior_head = PriorHead(ctx_dim=ctx_dim, v_dim=v_dim, prior_K=prior_K)

    def _rest_ctx(self, x_rest: torch.Tensor) -> torch.Tensor:
        x_rest_in = x_rest
        if self.hrf_kernel is not None:
            x_rest_in = self.hrf_kernel(x_rest)
        return self.rest_enc(x_rest_in)

    def sample_x0(self, x_rest: torch.Tensor, T: int) -> torch.Tensor:
        """
        Sample x0 from rest-conditioned prior with cross-ROI structure:
          mean, std, U = prior_head(ctx); z ~ N(0,1) (B,T,K); corr = z @ U^T;
          x0 = mean + std*eps_pink + corr.
        eps is drawn as pink (1/f) noise so the prior already carries low-frequency
        energy, reducing the distance the flow must travel in spectral space.
        Returns (B, T, V).
        """
        ctx = self._rest_ctx(x_rest)
        if self.use_prior_detach:
            ctx = ctx.detach()
        mean, std, U = self.prior_head(ctx)
        B, V, K = U.shape

        # --- pink-noise prior (1/f colouring) ---
        eps = torch.randn(B, T, V, device=x_rest.device, dtype=x_rest.dtype)
        freqs = torch.fft.rfftfreq(T, device=x_rest.device, dtype=x_rest.dtype)
        freqs[0] = freqs[1]                                   # avoid DC divide-by-zero
        pink_filter = (1.0 / freqs.sqrt())                    # (F,)
        pink_filter = pink_filter.unsqueeze(0).unsqueeze(-1)  # (1, F, 1) -> broadcast over (B, F, V)
        eps_f = torch.fft.rfft(eps, dim=1)                    # (B, F, V)
        eps_f = eps_f * pink_filter
        eps = torch.fft.irfft(eps_f, n=T, dim=1)             # (B, T, V)
        eps = eps / (eps.std(dim=1, keepdim=True) + 1e-8)    # restore unit variance per ROI
        # -----------------------------------------

        z = torch.randn(B, T, K, device=x_rest.device, dtype=x_rest.dtype)
        corr = z @ U.transpose(-1, -2)
        return mean.unsqueeze(1) + std.unsqueeze(1) * eps + corr

    def velocity(self, t, x_t, x_rest, ev=None, ev_mask=None, task_start_idx=None):
        ctx = self._rest_ctx(x_rest)
        te = self.t_emb(t)
        event_tokens = None
        if self.use_evs and ev is not None and self.ev_encoder is not None:
            event_tokens = self.ev_encoder(ev)
        return self.vnet(
            x_t, ctx, te,
            event_tokens=event_tokens,
            ev_mask=ev_mask,
            ev=ev,
            task_start_idx=task_start_idx,
        )

    def sample(self, x_rest, T_pred: int, steps: int = 50, ev=None, ev_mask=None, task_start_idx=None):
        """
        Encode rest -> sample x0 from prior -> integrate ODE to t=1.
        x_rest: (B,L,V); optional ev (B,N_events,4), ev_mask (B,N_events), task_start_idx (B,) for window-relative EV onsets.
        returns x1_hat: (B,T,V)
        """
        x = self.sample_x0(x_rest, T_pred)
        B = x_rest.shape[0]
        dt = 1.0 / steps
        for k in range(steps):
            t = torch.full((B,), k * dt, device=x_rest.device)
            v = self.velocity(t, x, x_rest, ev=ev, ev_mask=ev_mask, task_start_idx=task_start_idx)
            x = x + dt * v
        return x


# ======================================================================
# Training / Evaluation
# ======================================================================

# -----------------------
# Differentiable Frequency and FC losses (PyTorch)
# -----------------------

def psd_power_torch(x: torch.Tensor, dim_time: int = 1) -> torch.Tensor:
    """
    Compute power spectral density via FFT. Differentiable.
    x: (B, T, V) -> returns (B, F, V) where F = T//2 + 1.
    """
    # x: (B, T, V)
    X = torch.fft.rfft(x, dim=dim_time)  # (B, F, V) complex
    power = (X.real ** 2 + X.imag ** 2) / x.shape[dim_time]
    return power


def frequency_loss_torch(
    pred: torch.Tensor,
    target: torch.Tensor,
    dim_time: int = 1,
    eps: float = 1e-8,
    fs: float = 1.0 / 0.72,
    low_hz: float = 0.01,
    high_hz: float = 0.05,
) -> torch.Tensor:
    """
    MSE between log-PSD of pred and target, restricted to the frequency band
    [low_hz, high_hz] (default 0.01–0.05 Hz, the core BOLD slow-fluctuation band).

    Args:
        pred / target: (B, T, V) time series
        fs: sampling rate in Hz (default 1/0.72 ≈ 1.389 Hz for TR=0.72s)
        low_hz / high_hz: frequency band boundaries in Hz
    """
    T = pred.shape[dim_time]
    p_pred = psd_power_torch(pred, dim_time=dim_time)    # (B, F, V)
    p_tgt  = psd_power_torch(target, dim_time=dim_time)

    # Build frequency-bin mask for [low_hz, high_hz]
    freqs = torch.fft.rfftfreq(T, d=1.0 / fs, device=pred.device, dtype=pred.dtype)  # (F,)
    mask = ((freqs >= low_hz) & (freqs <= high_hz)).float()                           # (F,)
    mask = mask.unsqueeze(0).unsqueeze(-1)                                            # (1, F, 1)

    log_pred = torch.log(p_pred + eps)
    log_tgt  = torch.log(p_tgt + eps)
    diff = (log_pred - log_tgt) ** 2                       # (B, F, V)
    # Masked mean: only average over bins inside the band
    masked_diff = diff * mask
    n_bins = mask.sum().clamp(min=1.0)
    return masked_diff.sum() / (n_bins * pred.shape[0] * pred.shape[2])


def coherence_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Magnitude-squared coherence between all ROI pairs at each frequency bin.
    x, y: (B, T, V).
    Returns: (B, F, V, V) where entry [b, f, i, j] = |S_ij(f)|^2 / (S_ii(f) * S_jj(f)).
    """
    X = torch.fft.rfft(x, dim=1)   # (B, F, V) complex
    Y = torch.fft.rfft(y, dim=1)   # (B, F, V) complex

    # Cross-spectral density S_ij(f) = X_i(f) * conj(Y_j(f))
    # X: (B, F, V, 1), Y_conj: (B, F, 1, V) -> (B, F, V, V)
    S_xy = X.unsqueeze(-1) * Y.conj().unsqueeze(-2)              # (B, F, V, V)
    # Auto-spectral densities S_ii, S_jj
    S_xx = (X.real ** 2 + X.imag ** 2).unsqueeze(-1)             # (B, F, V, 1)
    S_yy = (Y.real ** 2 + Y.imag ** 2).unsqueeze(-2)             # (B, F, 1, V)

    # |S_ij|^2 / (S_ii * S_jj)
    coh = (S_xy.real ** 2 + S_xy.imag ** 2) / (S_xx * S_yy + eps)
    return coh                                                     # (B, F, V, V)


def coherence_loss_torch(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    L1 loss between coherence matrices of pred and target, summed over
    frequency bins and averaged over the upper-triangle ROI pairs.
    Scale-normalized and captures frequency-specific inter-ROI coupling
    that PSD alone ignores.

    pred / target: (B, T, V).
    """
    coh_pred = coherence_torch(pred, pred, eps=eps)     # (B, F, V, V)
    coh_tgt  = coherence_torch(target, target, eps=eps) # (B, F, V, V)

    B, F_bins, V, _ = coh_pred.shape
    # Upper triangle indices (exclude diagonal — self-coherence is always 1)
    triu_idx = torch.triu_indices(V, V, offset=1, device=pred.device)
    # Extract upper triangle: (B, F, N_pairs)
    coh_pred_ut = coh_pred[:, :, triu_idx[0], triu_idx[1]]
    coh_tgt_ut  = coh_tgt[:, :, triu_idx[0], triu_idx[1]]

    # L1 over frequencies, mean over batch and pairs
    return torch.abs(coh_pred_ut - coh_tgt_ut).mean()


def fc_matrix_torch(x: torch.Tensor, dim_time: int = 1, eps: float = 1e-8) -> torch.Tensor:
    """
    Correlation matrix (functional connectivity) over time. Differentiable.
    x: (B, T, V) -> (B, V, V).
    """
    x_centered = x - x.mean(dim=dim_time, keepdim=True)
    std = x_centered.std(dim=dim_time, keepdim=True) + eps
    x_norm = x_centered / std
    T = x.shape[dim_time]
    cov = torch.bmm(x_norm.transpose(1, 2), x_norm) / (T - 1)
    return torch.clamp(cov, -1.0, 1.0)


def fc_loss_torch(
    pred: torch.Tensor,
    target: torch.Tensor,
    use_upper_triangle: bool = True,
    weight_by_strength: bool = True,
    strength_power: float = 2.0,
) -> torch.Tensor:
    """
    Weighted MSE between FC matrices of pred and target.
    Strong hub connections are emphasised via |FC_target|^strength_power weighting.

    Args:
        pred / target: (B, T, V) time series
        use_upper_triangle: only use upper triangle to avoid redundancy
        weight_by_strength: upweight strong (hub) connections
        strength_power: exponent for strength weighting (default 2.0)
    """
    fc_pred = fc_matrix_torch(pred)
    fc_tgt  = fc_matrix_torch(target)

    if use_upper_triangle:
        B, V, _ = fc_pred.shape
        triu_idx = torch.triu_indices(V, V, offset=1, device=pred.device)
        fc_pred_flat = fc_pred[:, triu_idx[0], triu_idx[1]]  # (B, N)
        fc_tgt_flat  = fc_tgt[:, triu_idx[0], triu_idx[1]]
    else:
        B, V, _ = fc_pred.shape
        fc_pred_flat = fc_pred.reshape(B, -1)
        fc_tgt_flat  = fc_tgt.reshape(B, -1)

    squared_error = (fc_pred_flat - fc_tgt_flat) ** 2               # (B, N)

    if weight_by_strength:
        weights = (torch.abs(fc_tgt_flat) + 1e-8) ** strength_power  # (B, N)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8) * weights.shape[1]
        return (squared_error * weights).mean()

    return squared_error.mean()


def flow_matching_loss(
    model: FMTS,
    x_rest,
    x_task,
    ev=None,
    ev_mask=None,
    task_start_idx=None,
    fm_loss_weight: float = 1.0,
    freq_loss_weight: float = 0.0,
    fc_loss_weight: float = 0.0,
    coh_loss_weight: float = 0.0,
    aux_ode_steps: int = 10,
    fc_weight_by_strength: bool = True,
    fc_strength_power: float = 2.0,
    **kwargs,
):
    """
    Composite flow matching training loss.

    loss = fm_loss_weight * L_fm
         + freq_loss_weight * L_freq   (log-PSD in 0.01–0.05 Hz band)
         + fc_loss_weight * L_fc       (weighted FC MSE)
         + coh_loss_weight * L_coh     (coherence L1)

    Each auxiliary loss uses an independent x0 sample → separate gradient paths.

    Returns: (loss_total, loss_fm, loss_freq, loss_fc, loss_coh)  — last four detached.
    """
    B, T, V = x_task.shape
    x0 = model.sample_x0(x_rest, T)
    x1 = x_task
    t = torch.rand(B, device=x_task.device)
    t_view = t.view(B, 1, 1)

    x_t = (1.0 - t_view) * x0 + t_view * x1
    v_star = (x1 - x0)

    v_pred = model.velocity(t, x_t, x_rest, ev=ev, ev_mask=ev_mask, task_start_idx=task_start_idx)
    loss_fm = F.mse_loss(v_pred, v_star)

    has_aux = freq_loss_weight > 0 or fc_loss_weight > 0 or coh_loss_weight > 0
    if not has_aux:
        zero = torch.tensor(0.0, device=x_task.device)
        return fm_loss_weight * loss_fm, loss_fm.detach(), zero, zero, zero

    t_zero = torch.zeros(B, device=x_task.device)
    loss_total = fm_loss_weight * loss_fm
    loss_freq = torch.tensor(0.0, device=x_task.device)
    loss_fc   = torch.tensor(0.0, device=x_task.device)
    loss_coh  = torch.tensor(0.0, device=x_task.device)

    # --- Frequency (PSD) loss: independent x0 → own gradient path ---
    if freq_loss_weight > 0:
        x0_freq = model.sample_x0(x_rest, T)
        v_freq = model.velocity(t_zero, x0_freq, x_rest, ev=ev, ev_mask=ev_mask, task_start_idx=task_start_idx)
        x1_hat_freq = x0_freq + v_freq
        loss_freq = frequency_loss_torch(x1_hat_freq, x1)
        loss_total = loss_total + freq_loss_weight * loss_freq

    # --- FC loss: independent x0 → own gradient path ---
    if fc_loss_weight > 0:
        x0_fc = model.sample_x0(x_rest, T)
        v_fc = model.velocity(t_zero, x0_fc, x_rest, ev=ev, ev_mask=ev_mask, task_start_idx=task_start_idx)
        x1_hat_fc = x0_fc + v_fc
        loss_fc = fc_loss_torch(
            x1_hat_fc, x1,
            weight_by_strength=fc_weight_by_strength,
            strength_power=fc_strength_power,
        )
        loss_total = loss_total + fc_loss_weight * loss_fc

    # --- Coherence loss: independent x0 → own gradient path ---
    if coh_loss_weight > 0:
        x0_coh = model.sample_x0(x_rest, T)
        v_coh = model.velocity(t_zero, x0_coh, x_rest, ev=ev, ev_mask=ev_mask, task_start_idx=task_start_idx)
        x1_hat_coh = x0_coh + v_coh
        loss_coh = coherence_loss_torch(x1_hat_coh, x1)
        loss_total = loss_total + coh_loss_weight * loss_coh

    return loss_total, loss_fm.detach(), loss_freq.detach(), loss_fc.detach(), loss_coh.detach()


def compute_val_composite_loss(
    model: FMTS,
    loader,
    device,
    fm_loss_weight: float = 1.0,
    freq_loss_weight: float = 0.0,
    fc_loss_weight: float = 0.0,
    coh_loss_weight: float = 0.0,
    acf_loss_weight: float = 0.0,
    aux_ode_steps: int = 10,
    fc_weight_by_strength: bool = True,
    fc_strength_power: float = 2.0,
    **kwargs,
) -> float:
    """
    Average composite loss (FM + freq + FC + coh, with the same weights as training)
    over a validation loader. Runs under torch.no_grad().
    """
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            x_rest = batch["input"].to(device).float()
            x_task = batch["target"].to(device).float()
            ev = batch["ev"].to(device).float() if batch.get("ev") is not None else None
            ev_mask = batch["ev_mask"].to(device).float() if batch.get("ev_mask") is not None else None
            task_start_idx = batch.get("task_start_idx")
            if task_start_idx is not None and not isinstance(task_start_idx, torch.Tensor):
                task_start_idx = torch.tensor(task_start_idx, device=device, dtype=torch.float32)
            elif task_start_idx is not None:
                task_start_idx = task_start_idx.to(device)

            loss, _, _, _, _ = flow_matching_loss(
                model,
                x_rest,
                x_task,
                ev=ev,
                ev_mask=ev_mask,
                task_start_idx=task_start_idx,
                fm_loss_weight=fm_loss_weight,
                freq_loss_weight=freq_loss_weight,
                fc_loss_weight=fc_loss_weight,
                coh_loss_weight=coh_loss_weight,
                aux_ode_steps=aux_ode_steps,
                fc_weight_by_strength=fc_weight_by_strength,
                fc_strength_power=fc_strength_power,
            )
            total += float(loss.item())
            n += 1
    return total / max(n, 1)


def train_epoch(
    model,
    loader,
    opt,
    device,
    max_grad_norm=1.0,
    fm_loss_weight=1.0,
    freq_loss_weight=0.0,
    fc_loss_weight=0.0,
    coh_loss_weight=0.0,
    acf_loss_weight=0.0,
    aux_ode_steps=10,
    fc_weight_by_strength=True,
    fc_strength_power=2.0,
    **kwargs,
):
    model.train()
    total = 0.0
    total_fm = 0.0
    total_freq = 0.0
    total_fc = 0.0
    total_coh = 0.0
    n = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        x_rest = batch["input"].to(device).float()
        x_task = batch["target"].to(device).float()
        ev = batch["ev"].to(device).float() if batch.get("ev") is not None else None
        ev_mask = batch["ev_mask"].to(device).float() if batch.get("ev_mask") is not None else None
        task_start_idx = batch.get("task_start_idx")
        if task_start_idx is not None and not isinstance(task_start_idx, torch.Tensor):
            task_start_idx = torch.tensor(task_start_idx, device=device, dtype=torch.float32)
        elif task_start_idx is not None:
            task_start_idx = task_start_idx.to(device)

        opt.zero_grad()
        loss, lm, lf, lc, lcoh = flow_matching_loss(
            model,
            x_rest,
            x_task,
            ev=ev,
            ev_mask=ev_mask,
            task_start_idx=task_start_idx,
            fm_loss_weight=fm_loss_weight,
            freq_loss_weight=freq_loss_weight,
            fc_loss_weight=fc_loss_weight,
            coh_loss_weight=coh_loss_weight,
            aux_ode_steps=aux_ode_steps,
            fc_weight_by_strength=fc_weight_by_strength,
            fc_strength_power=fc_strength_power,
        )
        if torch.isnan(loss):
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()

        total      += float(loss.item())
        total_fm   += float(lm.item())
        total_freq += float(lf.item())
        total_fc   += float(lc.item())
        total_coh  += float(lcoh.item())
        n += 1
    n = max(n, 1)
    avg_total = total / n
    avg_fm    = total_fm / n
    avg_freq  = total_freq / n
    avg_fc    = total_fc / n
    avg_coh   = total_coh / n
    print(f"  [loss breakdown] total={avg_total:.4f}  fm={avg_fm:.4f}  freq={avg_freq:.4f}  fc={avg_fc:.4f}  coh={avg_coh:.4f}")
    return avg_total, avg_fm, avg_freq, avg_fc, avg_coh

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

def evaluate_subject_level_dedup(model, loader, device, pred_len, ode_steps=50, **kwargs):
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
            ev = batch["ev"].to(device).float() if batch.get("ev") is not None else None
            ev_mask = batch["ev_mask"].to(device).float() if batch.get("ev_mask") is not None else None

            if not isinstance(sids, (list, tuple)):
                sids = list(sids)
            if not isinstance(starts, (list, tuple)):
                starts = list(starts)

            task_start_idx = None
            if isinstance(starts, torch.Tensor):
                task_start_idx = starts.to(x_rest.device)
            elif starts is not None:
                task_start_idx = torch.tensor(starts, device=x_rest.device, dtype=torch.float32)
            x_pred = model.sample(
                x_rest, T_pred=pred_len, steps=ode_steps,
                ev=ev, ev_mask=ev_mask, task_start_idx=task_start_idx,
            )

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
        topk = compute_fc_topk_precision_recall_auc(pred_full, tgt_full)
        topk_fc_sim = compute_fc_topk_similarity(pred_full, tgt_full)

        per_subj[sid] = {
            "mse": mse,
            "mae": mae,
            "freq_diff": freq_diff,
            "fc_similarity": fc_sim,
            "n_timepoints": int(total_len),
            **{k: v for k, v in topk.items() if k != "k_percentiles" and isinstance(v, (int, float))},
            **{k: v for k, v in topk_fc_sim.items() if k != "k_percentiles" and isinstance(v, (int, float))},
        }

    # Average across subjects (top-k keys from default k_percentiles 5,10,20,50)
    keys = ["mse", "mae", "freq_diff", "fc_similarity"]
    k_pct = (5, 10, 20, 50)
    topk_keys = [f"precision_at_{k}" for k in k_pct] + [f"recall_at_{k}" for k in k_pct] + [f"auc_at_{k}" for k in k_pct]
    topk_fc_sim_keys = [f"fc_similarity_at_{k}" for k in k_pct]
    keys = keys + topk_keys + topk_fc_sim_keys
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

    im1 = axes[1].imshow(fc_pred, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
    axes[1].set_title("Generated FC")
    axes[1].set_xlabel("ROI")
    axes[1].set_ylabel("ROI")

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


def plot_time_series_gt_vs_pred(
    pred_full: np.ndarray,
    tgt_full: np.ndarray,
    save_path: str,
    title_prefix: str = "",
    num_rois_to_plot: int = 5,
    roi_indices: list = None,
    tr_sec: float = None,
):
    """
    Plot fMRI time series: ground truth vs FM-generated (paper-ready).
    pred_full, tgt_full: (T, V) time x ROIs.
    Saves one subplot per selected ROI with GT and predicted traces.
    """
    T, V = pred_full.shape
    if roi_indices is not None:
        roi_idx = np.asarray(roi_indices, dtype=int)
        roi_idx = roi_idx[(roi_idx >= 0) & (roi_idx < V)]
        if len(roi_idx) == 0:
            roi_idx = np.linspace(0, V - 1, min(num_rois_to_plot, V), dtype=int)
    else:
        roi_idx = np.linspace(0, V - 1, min(num_rois_to_plot, V), dtype=int)
    n_roi = len(roi_idx)

    time_axis = np.arange(T, dtype=float)
    if tr_sec is not None:
        time_axis = time_axis * tr_sec
        xlabel = "Time (s)"
    else:
        xlabel = "Time (TR)"

    fig, axes = plt.subplots(n_roi, 1, figsize=(10, 1.8 * n_roi), sharex=True)
    if n_roi == 1:
        axes = [axes]
    for i, v in enumerate(roi_idx):
        ax = axes[i]
        ax.plot(time_axis, tgt_full[:, v], "b-", linewidth=1.5, label="Ground truth", alpha=0.9)
        ax.plot(time_axis, pred_full[:, v], "r--", linewidth=1.5, label="FM-generated", alpha=0.9)
        corr = np.corrcoef(pred_full[:, v], tgt_full[:, v])[0, 1] if T > 1 else 0.0
        mse = float(np.mean((pred_full[:, v] - tgt_full[:, v]) ** 2))
        ax.set_ylabel("Signal")
        ax.set_title(f"ROI {v}  |  r = {corr:.3f}, MSE = {mse:.4f}")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel(xlabel)
    fig.suptitle(f"{title_prefix}Time series: GT vs generated", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_subject_level_dedup_with_best_subject(
    model, loader, device, pred_len, ode_steps=50,
    out_dir: str = None, fs: float = 0.72, **kwargs
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
            ev = batch["ev"].to(device).float() if batch.get("ev") is not None else None
            ev_mask = batch["ev_mask"].to(device).float() if batch.get("ev_mask") is not None else None

            if not isinstance(sids, (list, tuple)):
                sids = list(sids)
            if not isinstance(starts, (list, tuple)):
                starts = list(starts)

            task_start_idx = None
            if isinstance(starts, torch.Tensor):
                task_start_idx = starts.to(x_rest.device)
            elif starts is not None:
                task_start_idx = torch.tensor(starts, device=x_rest.device, dtype=torch.float32)
            x_pred = model.sample(
                x_rest, T_pred=pred_len, steps=ode_steps,
                ev=ev, ev_mask=ev_mask, task_start_idx=task_start_idx,
            )
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
        topk = compute_fc_topk_precision_recall_auc(pred_full, tgt_full)
        topk_fc_sim = compute_fc_topk_similarity(pred_full, tgt_full)

        per_subj[sid] = {
            "mse": mse, "mae": mae, "freq_diff": freq_diff, "fc_similarity": fc_sim,
            **{k: v for k, v in topk.items() if k != "k_percentiles" and isinstance(v, (int, float))},
            **{k: v for k, v in topk_fc_sim.items() if k != "k_percentiles" and isinstance(v, (int, float))},
        }

        if mse < best_mse:
            best_mse = mse
            best_sid = sid
            best_pair = (pred_full, tgt_full)

    # Aggregate metrics (include top-k precision/recall/AUC and top-k FC similarity)
    metrics = {"num_subjects": len(per_subj)}
    k_pct = (5, 10, 20, 50)
    topk_keys = [f"precision_at_{k}" for k in k_pct] + [f"recall_at_{k}" for k in k_pct] + [f"auc_at_{k}" for k in k_pct]
    topk_fc_sim_keys = [f"fc_similarity_at_{k}" for k in k_pct]
    keys = ["mse", "mae", "freq_diff", "fc_similarity"] + topk_keys + topk_fc_sim_keys
    if len(per_subj) == 0:
        for k in keys:
            metrics[k] = float("nan")
            metrics[k + "_std"] = float("nan")
        return metrics

    for k in keys:
        vals = np.array([per_subj[s].get(k, float("nan")) for s in per_subj], dtype=np.float64)
        metrics[k] = float(np.nanmean(vals))
        metrics[k + "_std"] = float(np.nanstd(vals))
    
    # Debug: verify top-k FC similarity metrics are present
    topk_fc_sim_debug = [f"fc_similarity_at_{k}" for k in k_pct]
    missing_fc_metrics = [k for k in topk_fc_sim_debug if k not in metrics or np.isnan(metrics.get(k, float("nan")))]
    if missing_fc_metrics:
        print(f"Warning: Missing top-k FC similarity metrics: {missing_fc_metrics}")
        print(f"Available keys in metrics: {[k for k in metrics.keys() if 'fc_similarity' in k]}")

    # Save plots for the closest subject
    if out_dir is not None and best_pair is not None:
        os.makedirs(out_dir, exist_ok=True)
        pred_full, tgt_full = best_pair

        fc_path = os.path.join(out_dir, f"best_subject_{best_sid}_fc_gt_vs_pred.pdf")
        psd_path = os.path.join(out_dir, f"best_subject_{best_sid}_psd_diff.png")
        ts_path = os.path.join(out_dir, f"best_subject_{best_sid}_time_series_gt_vs_pred.png")

        plot_fc_gt_vs_pred(pred_full, tgt_full, fc_path, title_prefix=f"FM-TS best subject {best_sid} | ")
        plot_psd_spectrum_difference(pred_full, tgt_full, psd_path, fs=fs)
        plot_time_series_gt_vs_pred(pred_full, tgt_full, ts_path, title_prefix=f"FM-TS best subject {best_sid} | ")

        # Save arrays for paper figure script (plot_paper_timeseries.py --run_dir <out_dir>)
        np.save(os.path.join(out_dir, "pred_full.npy"), pred_full)
        np.save(os.path.join(out_dir, "tgt_full.npy"), tgt_full)

        print(f"[Saved] FC plot:  {fc_path}")
        print(f"[Saved] PSD plot: {psd_path}")
        print(f"[Saved] Time series plot: {ts_path}")
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
    p.add_argument("--use_evs", action="store_true", help="Use EV data from extracted_txt/EVs/{TASK_NAME}/{subject_id}")
    p.add_argument("--ev_root", type=str, default=None, help="Base path for EVs (extracted_txt/EVs). Omit for auto (HCP if exists, else cwd). Pass e.g. . or repo path for local testing.")
    p.add_argument("--lookback_length", type=int, default=512)
    p.add_argument("--prediction_length", type=int, default=None)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--normalize", action="store_true", default=True)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)

    # Model
    p.add_argument("--rest_hidden", type=int, default=256)
    p.add_argument("--ctx_dim", type=int, default=256)
    p.add_argument("--rest_encoder", type=str, default="transformer", choices=("transformer", "lstm"), help="Rest encoder: transformer (patch-based) or lstm")
    p.add_argument("--rest_patch_len", type=int, default=16, help="Patch length for transformer rest encoder")
    p.add_argument("--rest_num_layers", type=int, default=2, help="Rest encoder layers (LSTM or transformer)")
    p.add_argument("--rest_nhead", type=int, default=4, help="Number of attention heads (transformer only)")
    p.add_argument("--rest_dim_feedforward", type=int, default=512, help="Transformer FFN dim (transformer only)")
    p.add_argument("--prior_K", type=int, default=8, help="Low-rank factor size for prior cross-ROI structure")
    p.add_argument("--use_prior_detach", action="store_true", help="Detach rest ctx before prior head to avoid prior gradients interfering with velocity conditioning")
    p.add_argument("--t_dim", type=int, default=128)
    p.add_argument("--num_conditions", type=int, default=32, help="Max condition code for EV embedding (use_evs)")
    p.add_argument("--d_ev", type=int, default=64, help="EV token dim and cross-attention dim (use_evs)")
    # Learnable HRF kernel (applied to rest before encoding)
    p.add_argument("--use_hrf_kernel", action="store_true", help="Apply learnable HRF-basis convolution to rest input")
    p.add_argument("--hrf_kernel_len", type=int, default=20, help="HRF kernel length in TRs")
    p.add_argument("--hrf_num_basis", type=int, default=3, help="Number of HRF basis functions (gamma + derivative + optional)")
    p.add_argument("--hrf_per_roi", action="store_true", help="Use a separate HRF kernel per ROI (else shared)")
    # Event HRF timecourse: per-event HRF weights → timecourse K/V for attention
    p.add_argument("--use_ev_hrf_timecourse", action="store_true", help="Use HRF timecourse-conditioned event K/V (requires --use_evs)")
    p.add_argument("--ev_hrf_kernel_len", type=int, default=20, help="HRF kernel length in TRs for event timecourse")
    p.add_argument("--ev_hrf_num_basis", type=int, default=3, help="Number of HRF basis per event")
    p.add_argument("--no_ev_hrf_delay_width", action="store_true", help="Disable per-event delay/width for HRF timecourse")
    p.add_argument("--ev_hrf_smooth_boxcar", action="store_true", help="Use smooth sigmoid boxcar instead of hard step")
    p.add_argument("--ev_hrf_boxcar_sigma", type=float, default=0.5, help="Smoothing sigma for smooth boxcar (TRs)")

    # Training
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--ode_steps", type=int, default=50)
    # Auxiliary losses (Frequency + FC)
    p.add_argument("--freq_loss_weight", type=float, default=0.1, help="Weight for PSD/frequency loss")
    p.add_argument("--fc_loss_weight", type=float, default=0.1, help="Weight for FC (correlation matrix) loss")
    p.add_argument("--fc_weight_by_strength", action="store_true", default=True, help="Weight FC loss by connection strength (emphasizes prominent correlations, default: True)")
    p.add_argument("--fc_no_weight_by_strength", action="store_false", dest="fc_weight_by_strength", help="Disable FC loss weighting by strength")
    p.add_argument("--fc_strength_power", type=float, default=2.0, help="Power for FC strength weighting (higher = more emphasis on strong connections, default: 2.0)")
    p.add_argument("--coh_loss_weight", type=float, default=0.0, help="Weight for coherence loss (frequency-specific inter-ROI coupling)")
    p.add_argument("--aux_ode_steps", type=int, default=10, help="ODE steps for auxiliary loss (x1_pred)")

    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--save_dir", type=str, default="./checkpoints_fmts")
    p.add_argument("--load_dir", type=str, default=None, help="If set, load best checkpoint from this dir (e.g. best run by FC from report_best_run.py) and run test eval only (no training).")

    args = p.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = args.load_dir if args.load_dir else args.save_dir
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Device: {args.device}")

    ds = HCPRestingFCDataset(
        data_root=args.data_root,
        task_root=args.task_root,
        task_name=args.task_name,
        use_evs=args.use_evs,
        ev_root=args.ev_root,
    )
    print(f"Subjects: {len(ds)}")
    if len(ds) == 0:
        raise SystemExit(
            "No subjects in dataset. With use_evs=True, subjects must have: (1) EV dir under ev_root/extracted_txt/EVs/<TASK>/<subject_id>/ "
            "with Sync.txt and condition .txt files, (2) rest data at data_root/<subject_id>/timeseries/REST1_LR_AAL3_ts.npy, "
            "(3) task data at ./data/palmer_scratch/<task_name>/<subject_id>/<subject_id>_AAL3_ts.npy. "
            "Check paths (--data_root, --ev_root) and that those files exist."
        )

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

    print(f"Train/val/test windows: {len(train_ds)} / {len(val_ds)} / {len(test_ds)}")
    if len(train_ds) == 0:
        raise SystemExit(
            "Training set has 0 windows. Each subject needs rest length >= lookback_length ({}), "
            "task length >= prediction_length ({}), and (if use_evs) loadable EV. "
            "Try smaller --lookback_length or --stride, or check that rest/task/EV paths exist for your subjects."
            .format(args.lookback_length, args.prediction_length)
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Best checkpoint by lowest validation MSE
    best_val_mse = float("inf")
    best_path = os.path.join(ckpt_dir, "best_fmts.pth")

    # If loading from checkpoint, load args first to match architecture
    if args.load_dir is not None:
        if not os.path.isfile(best_path):
            raise SystemExit(f"Checkpoint not found: {best_path}. Use --load_dir with a dir that contains best_fmts.pth (e.g. best run from report_best_run.py).")
        print(f"Loading checkpoint args from {best_path} to match model architecture...")
        ckpt = torch.load(best_path, map_location=args.device)
        
        # If checkpoint has saved args, use them to ensure model architecture matches
        if "args" in ckpt and ckpt["args"] is not None:
            saved_args = ckpt["args"]
            print("Found saved args in checkpoint. Using saved model architecture.")
            # Override args with saved ones for model architecture
            for key in ["use_evs", "use_hrf_kernel", "use_ev_hrf_timecourse", 
                       "rest_encoder", "rest_hidden", "ctx_dim", "t_dim",
                       "rest_patch_len", "rest_num_layers", "rest_nhead", 
                       "rest_dim_feedforward", "prior_K", "use_prior_detach",
                       "num_conditions", "d_ev", "hrf_kernel_len", "hrf_num_basis",
                       "hrf_per_roi", "ev_hrf_kernel_len", "ev_hrf_num_basis",
                       "no_ev_hrf_delay_width", "ev_hrf_smooth_boxcar", "ev_hrf_boxcar_sigma"]:
                if key in saved_args:
                    setattr(args, key, saved_args[key])
        else:
            print("Warning: Checkpoint does not contain saved args. Using current args (may cause mismatch).")

    # infer V
    sample = next(iter(train_loader))
    V = int(sample["input"].shape[-1])
    print(f"V (ROIs) = {V}")

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
        ev_hrf_use_delay_width=not args.no_ev_hrf_delay_width,
        ev_hrf_smooth_boxcar=args.ev_hrf_smooth_boxcar,
        ev_hrf_boxcar_sigma=args.ev_hrf_boxcar_sigma,
        prior_K=args.prior_K,
        use_prior_detach=args.use_prior_detach,
    ).to(args.device)
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)

    if args.load_dir is not None:
        # Eval-only: load best run from load_dir and run test
        print(f"Loading model weights from {best_path} (eval-only, no training).")
        # Load state dict with error handling
        try:
            model.load_state_dict(ckpt["model"], strict=True)
            print("Successfully loaded checkpoint (strict mode).")
        except RuntimeError as e:
            print(f"Warning: Strict loading failed: {e}")
            print("Attempting non-strict loading (ignoring missing/extra keys)...")
            missing_keys, unexpected_keys = model.load_state_dict(ckpt["model"], strict=False)
            if missing_keys:
                print(f"Missing keys (not loaded): {len(missing_keys)} keys")
                if len(missing_keys) <= 20:
                    print(f"  {missing_keys}")
                else:
                    print(f"  First 10: {missing_keys[:10]}")
            if unexpected_keys:
                print(f"Unexpected keys (ignored): {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 20:
                    print(f"  {unexpected_keys}")
                else:
                    print(f"  First 10: {unexpected_keys[:10]}")
            if missing_keys and len(missing_keys) > 0:
                print("Warning: Some model parameters were not loaded. Results may be incorrect.")
    else:
        for ep in range(1, args.epochs + 1):
            tr_loss, tr_fm, tr_freq, tr_fc, tr_coh = train_epoch(
                model,
                train_loader,
                opt,
                args.device,
                max_grad_norm=args.max_grad_norm,
                freq_loss_weight=args.freq_loss_weight,
                fc_loss_weight=args.fc_loss_weight,
                aux_ode_steps=args.aux_ode_steps,
                coh_loss_weight=args.coh_loss_weight,
                fc_weight_by_strength=args.fc_weight_by_strength,
                fc_strength_power=args.fc_strength_power,
            )

            # subject-level dedup validation
            val_metrics = evaluate_subject_level_dedup(
                model, val_loader, args.device,
                pred_len=args.prediction_length,
                ode_steps=args.ode_steps
            )
            sched.step()

            print(f"\nEpoch {ep}/{args.epochs}")
            print(f"  Train total={tr_loss:.6f}  fm={tr_fm:.6f}  freq={tr_freq:.6f}  fc={tr_fc:.6f}  coh={tr_coh:.6f}")
            print(f"  Val (subject-avg) MSE: {val_metrics['mse']:.6f} ± {val_metrics['mse_std']:.6f}")
            print(f"  Val (subject-avg) MAE: {val_metrics['mae']:.6f} ± {val_metrics['mae_std']:.6f}")
            print(f"  Val (subject-avg) PSD: {val_metrics['freq_diff']:.6f} ± {val_metrics['freq_diff_std']:.6f}")
            print(f"  Val (subject-avg) FC : {val_metrics['fc_similarity']:.6f} ± {val_metrics['fc_similarity_std']:.6f}")
            print(f"  Subjects: {val_metrics['num_subjects']}")

            if not math.isnan(val_metrics['mse']) and val_metrics['mse'] < best_val_mse:
                best_val_mse = val_metrics['mse']
                torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)
                print(f"  Saved best (val_mse={best_val_mse:.6f}) -> {best_path}")

    # Test (load from checkpoint if we trained; already loaded if --load_dir)
    if args.load_dir is None:
        if not os.path.isfile(best_path):
            print(f"WARNING: No best checkpoint found at {best_path} (training likely diverged to NaN). Skipping test eval.")
            return
        ckpt = torch.load(best_path, map_location=args.device)
        model.load_state_dict(ckpt["model"])
    test_metrics = evaluate_subject_level_dedup_with_best_subject(
        model, test_loader, args.device,
        pred_len=args.prediction_length,
        ode_steps=args.ode_steps,
        out_dir=args.save_dir,
        fs=0.72,
    )

    # cFID-FC: conditional Fréchet distance on FC features (paired real vs generated)
    cfid_fc_value = None
    try:
        _repo_root = Path(__file__).resolve().parent.parent
        _re_eval = _repo_root / "re_eval"
        if _re_eval.exists():
            sys.path.insert(0, str(_re_eval))
            from fc_utils import cfid_fc as _cfid_fc
            real_list, gen_list = [], []
            with torch.no_grad():
                for batch in test_loader:
                    x_rest = batch["input"].to(args.device).float()
                    x_task = batch["target"].to(args.device).float()
                    starts = batch["task_start_idx"]
                    ev = batch["ev"].to(args.device).float() if batch.get("ev") is not None else None
                    ev_mask = batch["ev_mask"].to(args.device).float() if batch.get("ev_mask") is not None else None
                    task_start_idx = starts.to(args.device) if isinstance(starts, torch.Tensor) else torch.tensor(starts, device=args.device, dtype=torch.float32)
                    x_pred = model.sample(
                        x_rest, T_pred=args.prediction_length, steps=args.ode_steps,
                        ev=ev, ev_mask=ev_mask, task_start_idx=task_start_idx,
                    )
                    real_list.append(x_task.cpu().numpy())
                    gen_list.append(x_pred.cpu().numpy())
            X_real = np.concatenate(real_list, axis=0)
            X_gen = np.concatenate(gen_list, axis=0)
            rng = np.random.default_rng(42)
            cfid_fc_value = _cfid_fc(X_real, X_gen, eps=1e-6, max_fc_dim=500, rng=rng)
            test_metrics["cfid_fc"] = cfid_fc_value
    except Exception as e:
        print(f"[cFID-FC] Skipped: {e}")

    k_pct = (5, 10, 20, 50)
    print("\n" + "=" * 70)
    print("FM-TS (Conditional Flow Matching) TEST (Subject-level, Dedup)")
    print("=" * 70)
    print(f"MSE (mean ± std): {test_metrics['mse']:.6f} ± {test_metrics['mse_std']:.6f}")
    print(f"MAE (mean ± std): {test_metrics['mae']:.6f} ± {test_metrics['mae_std']:.6f}")
    print(f"PSD MAE (mean ± std): {test_metrics['freq_diff']:.6f} ± {test_metrics['freq_diff_std']:.6f}")
    print(f"FC sim (mean ± std): {test_metrics['fc_similarity']:.6f} ± {test_metrics['fc_similarity_std']:.6f}")
    print("Top-k connectivity (precision / recall / AUC by top k% of edges):")
    for k in k_pct:
        pk = test_metrics.get(f"precision_at_{k}", float("nan"))
        rk = test_metrics.get(f"recall_at_{k}", float("nan"))
        ak = test_metrics.get(f"auc_at_{k}", float("nan"))
        pk_std = test_metrics.get(f"precision_at_{k}_std", 0.0)
        rk_std = test_metrics.get(f"recall_at_{k}_std", 0.0)
        ak_std = test_metrics.get(f"auc_at_{k}_std", 0.0)
        print(f"  k={k}%:  Precision {pk:.4f} ± {pk_std:.4f}   Recall {rk:.4f} ± {rk_std:.4f}   AUC {ak:.4f} ± {ak_std:.4f}")
    print("Top-k FC similarity (Pearson correlation on top k% connections):")
    has_topk_fc = False
    for k in k_pct:
        fc_sim_k = test_metrics.get(f"fc_similarity_at_{k}", float("nan"))
        fc_sim_k_std = test_metrics.get(f"fc_similarity_at_{k}_std", 0.0)
        if not np.isnan(fc_sim_k):
            has_topk_fc = True
        print(f"  k={k}%:  FC similarity {fc_sim_k:.4f} ± {fc_sim_k_std:.4f}")
    if not has_topk_fc:
        print("  Warning: No top-k FC similarity metrics found in results. Check that compute_fc_topk_similarity is being called.")
    print(f"Num subjects: {test_metrics['num_subjects']}")
    if cfid_fc_value is not None:
        print(f"cFID-FC (conditional Fréchet on FC): {cfid_fc_value:.6f}  (lower = better)")
    print("=" * 70)

    test_results_path = os.path.join(args.save_dir, "test_results.txt")
    with open(test_results_path, "w", encoding="utf-8") as f:
        f.write("FM-TS TEST (Subject-level, Dedup)\n")
        f.write("=" * 70 + "\n")
        f.write(f"MSE (mean ± std): {test_metrics['mse']:.6f} ± {test_metrics['mse_std']:.6f}\n")
        f.write(f"MAE (mean ± std): {test_metrics['mae']:.6f} ± {test_metrics['mae_std']:.6f}\n")
        f.write(f"PSD MAE (mean ± std): {test_metrics['freq_diff']:.6f} ± {test_metrics['freq_diff_std']:.6f}\n")
        f.write(f"FC sim (mean ± std): {test_metrics['fc_similarity']:.6f} ± {test_metrics['fc_similarity_std']:.6f}\n")
        f.write("Top-k connectivity (precision / recall / AUC by top k% of edges):\n")
        for k in k_pct:
            pk = test_metrics.get(f"precision_at_{k}", float("nan"))
            rk = test_metrics.get(f"recall_at_{k}", float("nan"))
            ak = test_metrics.get(f"auc_at_{k}", float("nan"))
            pk_std = test_metrics.get(f"precision_at_{k}_std", 0.0)
            rk_std = test_metrics.get(f"recall_at_{k}_std", 0.0)
            ak_std = test_metrics.get(f"auc_at_{k}_std", 0.0)
            f.write(f"  k={k}%:  Precision {pk:.4f} ± {pk_std:.4f}   Recall {rk:.4f} ± {rk_std:.4f}   AUC {ak:.4f} ± {ak_std:.4f}\n")
        f.write("Top-k FC similarity (Pearson correlation on top k% connections):\n")
        has_topk_fc = False
        for k in k_pct:
            fc_sim_k = test_metrics.get(f"fc_similarity_at_{k}", float("nan"))
            fc_sim_k_std = test_metrics.get(f"fc_similarity_at_{k}_std", 0.0)
            if not np.isnan(fc_sim_k):
                has_topk_fc = True
            f.write(f"  k={k}%:  FC similarity {fc_sim_k:.4f} ± {fc_sim_k_std:.4f}\n")
        if not has_topk_fc:
            f.write("  Warning: No top-k FC similarity metrics found in results.\n")
        f.write(f"Num subjects: {test_metrics['num_subjects']}\n")
        if cfid_fc_value is not None:
            f.write(f"cFID-FC (conditional Fréchet on FC): {cfid_fc_value:.6f}  (lower = better)\n")
        f.write("=" * 70 + "\n")
    print(f"Test results written to {test_results_path}")


if __name__ == "__main__":
    main()
