"""
DDPM (Denoising Diffusion Probabilistic Models) Baseline for Rest-to-Task fMRI Prediction

Conditional DDPM for time series: rest-conditioned noise prediction.
- Forward: q(x_t | x_{t-1}) with linear beta schedule
- Train: predict epsilon given (x_t, t, rest_ctx) with MSE (Ho et al.)
- Sample: reverse process x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps_pred) + sigma_t * z

Subject-level evaluation: aggregate predicted windows per subject, deduplicate, then MSE/MAE/FC/PSD.
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

_project_root = Path(__file__).resolve().parent.parent
_baselines_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_baselines_dir))
from dataset import HCPRestingFCDataset
from eval_viz import save_closest_subject_visualizations
from aux_losses import frequency_loss_torch
import subprocess


def _maybe_compute_cfid(save_dir, args):
    """
    Compute cFID-FC for this DDPM checkpoint by calling the separate re_eval script
    in a subprocess. Returns float value or None on failure.
    """
    cfid_script = _project_root / "re_eval" / "run_cfid_baseline.py"
    if not cfid_script.exists():
        print("[cFID-FC] re_eval/run_cfid_baseline.py not found; skipping cFID.", flush=True)
        return None

    cmd = [
        sys.executable,
        str(cfid_script),
        "--load_dir",
        str(save_dir),
        "--model_type",
        "ddpm",
        "--task_name",
        str(args.task_name),
        "--data_root",
        str(args.data_root),
        "--task_root",
        str(args.task_root),
        "--sample_steps",
        str(args.sample_steps),
    ]

    try:
        subprocess.run(cmd, check=True)
        cfid_path = Path(save_dir) / "cfid_fc.txt"
        if not cfid_path.exists():
            return None
        text = cfid_path.read_text(encoding="utf-8").strip()
        first_line = text.splitlines()[0] if text else ""
        parts = first_line.strip().split()
        if len(parts) >= 2:
            return float(parts[1])
    except Exception as e:
        print(f"  [cFID-FC] ddpm failed: {e}", flush=True)
    return None


# ======================================================================
# Dataset (same as FM-TS / Diffusion-TS)
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
        self.rest_means = self.rest_stds = self.task_means = self.task_stds = None
        self._create_window_indices(train_ratio, val_ratio)
        if self.normalize and len(self.window_metadata) > 0:
            self._compute_normalization_stats(sample_size=norm_sample_size, batch_size=norm_batch_size)

    def _create_window_indices(self, train_ratio: float, val_ratio: float):
        all_subjects = self.dataset.subject_ids
        n = len(all_subjects)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        subject_ids = all_subjects[:train_end] if self.split == "train" else \
                      all_subjects[train_end:val_end] if self.split == "val" else all_subjects[val_end:]
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
                max_windows = min(R - self.lookback_length, T - self.prediction_length)
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
        rest_sum = rest_sum_sq = task_sum = task_sum_sq = None
        rest_cnt = task_cnt = 0
        V = None
        for s in range(0, len(idxs), batch_size):
            batch_idxs = idxs[s : s + batch_size]
            rest_batch, task_batch = [], []
            for ii in batch_idxs:
                meta = self.window_metadata[ii]
                sid = meta["subject_id"]
                rest = self.dataset.load_subject(sid)
                if rest.ndim == 1:
                    rest = rest.reshape(-1, 1)
                if V is None:
                    V = rest.shape[1]
                rs, re = meta["rest_start_idx"], meta["rest_start_idx"] + self.lookback_length
                rest_batch.append(rest[rs:re].astype(np.float32))
                task = self.dataset.load_task_subject(sid)
                if task.ndim == 1:
                    task = task.reshape(-1, 1)
                ts, te = meta["task_start_idx"], meta["task_start_idx"] + self.prediction_length
                task_batch.append(task[ts:te].astype(np.float32))
            if rest_batch:
                r = np.stack(rest_batch).reshape(-1, V)
                rest_sum = r.sum(0) if rest_sum is None else rest_sum + r.sum(0)
                rest_sum_sq = (r**2).sum(0) if rest_sum_sq is None else rest_sum_sq + (r**2).sum(0)
                rest_cnt += r.shape[0]
            if task_batch:
                t = np.stack(task_batch).reshape(-1, V)
                task_sum = t.sum(0) if task_sum is None else task_sum + t.sum(0)
                task_sum_sq = (t**2).sum(0) if task_sum_sq is None else task_sum_sq + (t**2).sum(0)
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
        rs, re = meta["rest_start_idx"], meta["rest_start_idx"] + self.lookback_length
        x = rest[rs:re].astype(np.float32)
        task = self.dataset.load_task_subject(sid)
        if task.ndim == 1:
            task = task.reshape(-1, 1)
        ts, te = meta["task_start_idx"], meta["task_start_idx"] + self.prediction_length
        y = task[ts:te].astype(np.float32)
        if self.normalize and self.rest_means is not None:
            x = (x - self.rest_means) / self.rest_stds
            y = (y - self.task_means) / self.task_stds
        return {"input": torch.from_numpy(x), "target": torch.from_numpy(y), "subject_id": sid, "task_start_idx": int(ts)}


# ======================================================================
# Metrics
# ======================================================================

def compute_functional_connectivity(data: np.ndarray) -> np.ndarray:
    fc = np.corrcoef(data.T)
    return np.nan_to_num(fc, nan=0.0, posinf=1.0, neginf=-1.0)

def compute_fc_similarity(pred: np.ndarray, target: np.ndarray) -> float:
    fc_pred = compute_functional_connectivity(pred)
    fc_tgt = compute_functional_connectivity(target)
    mask = np.triu(np.ones_like(fc_pred, dtype=bool), k=1)
    a, b = fc_pred[mask], fc_tgt[mask]
    if len(a) > 1 and np.std(a) > 1e-10 and np.std(b) > 1e-10:
        r, _ = pearsonr(a, b)
        return float(r) if not np.isnan(r) else 0.0
    return 0.0


# Top-k connectivity metrics (precision/recall/AUC at 5%, 10%, 20%, 50%)
from fc_metrics import compute_fc_topk_precision_recall_auc, topk_metric_keys, TOP_K_PERCENTILES


def _print_topk(metrics):
    for k in TOP_K_PERCENTILES:
        pk = metrics.get(f"precision_at_{k}", float("nan"))
        rk = metrics.get(f"recall_at_{k}", float("nan"))
        ak = metrics.get(f"auc_at_{k}", float("nan"))
        pk_std = metrics.get(f"precision_at_{k}_std", 0.0)
        rk_std = metrics.get(f"recall_at_{k}_std", 0.0)
        ak_std = metrics.get(f"auc_at_{k}_std", 0.0)
        print(f"  k={k}%:  Precision {pk:.4f} ± {pk_std:.4f}   Recall {rk:.4f} ± {rk_std:.4f}   AUC {ak:.4f} ± {ak_std:.4f}")


def _write_topk(f, metrics):
    f.write("Top-k connectivity (precision / recall / AUC by top k% of edges):\n")
    for k in TOP_K_PERCENTILES:
        pk = metrics.get(f"precision_at_{k}", float("nan"))
        rk = metrics.get(f"recall_at_{k}", float("nan"))
        ak = metrics.get(f"auc_at_{k}", float("nan"))
        pk_std = metrics.get(f"precision_at_{k}_std", 0.0)
        rk_std = metrics.get(f"recall_at_{k}_std", 0.0)
        ak_std = metrics.get(f"auc_at_{k}_std", 0.0)
        f.write(f"  k={k}%:  Precision {pk:.4f} ± {pk_std:.4f}   Recall {rk:.4f} ± {rk_std:.4f}   AUC {ak:.4f} ± {ak_std:.4f}\n")

def compute_frequency_difference(pred: np.ndarray, target: np.ndarray, fs: float = 0.72) -> float:
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

def aggregate_subject_timeline(chunks, starts, total_len):
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


# ======================================================================
# DDPM: rest encoder + time embedding + epsilon prediction net
# ======================================================================

class RestEncoder(nn.Module):
    def __init__(self, v_dim: int, hidden: int = 256, layers: int = 2, out_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(v_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.proj(h[:, -1, :])

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000.0) * torch.arange(half, device=t.device).float() / half)
        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class EpsilonNet(nn.Module):
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
        ctx = rest_ctx.unsqueeze(1).expand(B, T, -1)
        te = t_emb.unsqueeze(1).expand(B, T, -1)
        inp = torch.cat([x_t, ctx, te], dim=-1)
        return self.net(inp)

class DDPM(nn.Module):
    """Conditional DDPM for rest-to-task time series (Ho et al. formulation)."""
    def __init__(self, v_dim: int, num_timesteps: int = 1000, rest_hidden: int = 256, ctx_dim: int = 256, t_dim: int = 128, beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        self.v_dim = v_dim
        self.num_timesteps = num_timesteps
        self.rest_enc = RestEncoder(v_dim=v_dim, hidden=rest_hidden, out_dim=ctx_dim)
        self.t_embed = nn.Sequential(
            SinusoidalTimeEmbedding(t_dim),
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )
        self.eps_net = EpsilonNet(v_dim=v_dim, ctx_dim=ctx_dim, t_dim=t_dim)
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))
        self.register_buffer("sqrt_recip_alpha", torch.sqrt(1.0 / alphas))
        self.register_buffer("sigma", torch.sqrt(betas))

    def forward_eps(self, t: torch.LongTensor, x_t: torch.Tensor, x_rest: torch.Tensor):
        rest_ctx = self.rest_enc(x_rest)
        t_emb = self.t_embed(t.float())
        return self.eps_net(x_t, rest_ctx, t_emb)

    def train_step_loss(self, x_0: torch.Tensor, x_rest: torch.Tensor):
        B, T, V = x_0.shape
        t = torch.randint(0, self.num_timesteps, (B,), device=x_0.device, dtype=torch.long)
        eps = torch.randn_like(x_0)
        sqrt_ab = self.sqrt_alpha_bar[t].view(B, 1, 1)
        sqrt_1_ab = self.sqrt_one_minus_alpha_bar[t].view(B, 1, 1)
        x_t = sqrt_ab * x_0 + sqrt_1_ab * eps
        eps_pred = self.forward_eps(t, x_t, x_rest)
        return F.mse_loss(eps_pred, eps)

    @torch.no_grad()
    def sample(self, x_rest: torch.Tensor, T_pred: int, num_steps: int = 50):
        B, L, V = x_rest.shape
        x = torch.randn(B, T_pred, V, device=x_rest.device)
        step = max(1, self.num_timesteps // num_steps)
        # Build indices so we always include t=0 (final denoised x_0). range(999,-1,-20) stops at 19!
        indices = list(range(self.num_timesteps - 1, -1, -step))
        if indices[-1] != 0:
            indices.append(0)
        clip_min, clip_max = -4.0, 4.0  # normalized data range; stabilizes and matches Diffusion-TS
        for k, i in enumerate(indices):
            t_batch = torch.full((B,), i, device=x_rest.device, dtype=torch.long)
            eps_pred = self.forward_eps(t_batch, x, x_rest)
            sqrt_recip_alpha = self.sqrt_recip_alpha[i].view(1, 1, 1)
            sqrt_1_ab = self.sqrt_one_minus_alpha_bar[i].view(1, 1, 1)
            # (1 - alpha_t) / (1 - alpha_bar_t) for posterior coef
            one_minus_alpha_bar = (1.0 - self.alpha_bar[i]).clamp(min=1e-8)
            coef = ((1.0 - self.alphas[i]) / one_minus_alpha_bar).sqrt().view(1, 1, 1)
            mean = sqrt_recip_alpha * (x - coef * eps_pred)
            if i == 0:
                x = mean.clamp(clip_min, clip_max)
                break
            sigma = self.sigma[i].view(1, 1, 1)
            x = (mean + sigma * torch.randn_like(x, device=x.device)).clamp(clip_min, clip_max)
        return x

    def sample_deterministic(self, x_rest: torch.Tensor, T_pred: int, num_steps: int = 10):
        """Deterministic reverse (no noise). Differentiable, for PSD auxiliary loss."""
        B, L, V = x_rest.shape
        x = torch.randn(B, T_pred, V, device=x_rest.device)
        step = max(1, self.num_timesteps // num_steps)
        indices = list(range(self.num_timesteps - 1, -1, -step))
        if indices[-1] != 0:
            indices.append(0)
        clip_min, clip_max = -4.0, 4.0
        for k, i in enumerate(indices):
            t_batch = torch.full((B,), i, device=x_rest.device, dtype=torch.long)
            eps_pred = self.forward_eps(t_batch, x, x_rest)
            sqrt_recip_alpha = self.sqrt_recip_alpha[i].view(1, 1, 1)
            one_minus_alpha_bar = (1.0 - self.alpha_bar[i]).clamp(min=1e-8)
            coef = ((1.0 - self.alphas[i]) / one_minus_alpha_bar).sqrt().view(1, 1, 1)
            mean = sqrt_recip_alpha * (x - coef * eps_pred)
            x = mean.clamp(clip_min, clip_max)
            if i == 0:
                break
        return x


# ======================================================================
# Train / Eval
# ======================================================================

def train_epoch(model, loader, opt, device, max_grad_norm=1.0, pred_len=None, freq_loss_weight=0.0, freq_aux_steps=10):
    model.train()
    total, n = 0.0, 0
    for batch in tqdm(loader, desc="Train", leave=False):
        x_rest = batch["input"].to(device).float()
        x_task = batch["target"].to(device).float()
        opt.zero_grad()
        loss = model.train_step_loss(x_task, x_rest)
        if freq_loss_weight > 0 and pred_len is not None:
            x_pred = model.sample_deterministic(x_rest, T_pred=pred_len, num_steps=freq_aux_steps)
            loss = loss + freq_loss_weight * frequency_loss_torch(x_pred, x_task)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()
        total += float(loss.item())
        n += 1
    return total / max(n, 1)

def evaluate_subject_level_dedup(model, loader, device, pred_len: int, num_steps: int = 50, return_for_viz: bool = False):
    model.eval()
    subj_pred = defaultdict(list)
    subj_tgt = defaultdict(list)
    subj_starts = defaultdict(list)
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
            x_pred = model.sample(x_rest, T_pred=pred_len, num_steps=num_steps)
            x_pred_np = x_pred.cpu().numpy()
            x_task_np = x_task.cpu().numpy()
            for b, sid in enumerate(sids):
                sid = str(sid)
                st = int(starts[b])
                subj_pred[sid].append(x_pred_np[b])
                subj_tgt[sid].append(x_task_np[b])
                subj_starts[sid].append(st)
                subj_total_len[sid] = max(subj_total_len[sid], st + pred_len)
    per_subj = {}
    for sid in sorted(subj_pred.keys()):
        total_len = subj_total_len[sid]
        pred_full = aggregate_subject_timeline(subj_pred[sid], subj_starts[sid], total_len)
        tgt_full = aggregate_subject_timeline(subj_tgt[sid], subj_starts[sid], total_len)
        mse = float(np.mean((pred_full - tgt_full) ** 2))
        mae = float(np.mean(np.abs(pred_full - tgt_full)))
        freq_diff = float(compute_frequency_difference(pred_full, tgt_full))
        fc_sim = float(compute_fc_similarity(pred_full, tgt_full))
        topk = compute_fc_topk_precision_recall_auc(pred_full, tgt_full)
        per_subj[sid] = {
            "mse": mse, "mae": mae, "freq_diff": freq_diff, "fc_similarity": fc_sim,
            **{key: val for key, val in topk.items() if key != "k_percentiles" and isinstance(val, (int, float))},
        }
    metrics = {"num_subjects": len(per_subj)}
    keys = ["mse", "mae", "freq_diff", "fc_similarity"] + topk_metric_keys()
    if len(per_subj) == 0:
        for k in keys:
            metrics[k] = float("nan")
            metrics[k + "_std"] = float("nan")
        return (metrics, subj_pred, subj_tgt, subj_starts, subj_total_len, per_subj) if return_for_viz else metrics
    for k in keys:
        vals = np.array([per_subj[s].get(k, float("nan")) for s in per_subj], dtype=np.float64)
        metrics[k] = float(np.nanmean(vals))
        metrics[k + "_std"] = float(np.nanstd(vals))
    if return_for_viz:
        return metrics, subj_pred, subj_tgt, subj_starts, subj_total_len, per_subj
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data/hcp-resting-fc")
    p.add_argument("--task_root", type=str, default="./data/hcp-task-ts")
    p.add_argument("--task_name", type=str, default="emotion")
    p.add_argument("--lookback_length", type=int, default=512)
    p.add_argument("--prediction_length", type=int, default=None)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--normalize", action="store_true", default=True)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--rest_hidden", type=int, default=256)
    p.add_argument("--ctx_dim", type=int, default=256)
    p.add_argument("--num_timesteps", type=int, default=1000)
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=0.02)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--freq_loss_weight", type=float, default=0.0, help="Weight for PSD auxiliary loss")
    p.add_argument("--freq_aux_steps", type=int, default=10, help="Reverse steps for PSD loss (deterministic)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--save_dir", type=str, default="./checkpoints_ddpm")
    p.add_argument("--eval_only", action="store_true", help="Load best checkpoint, run test eval and save FC/PSD visualizations only")
    args = p.parse_args()

    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    ds = HCPRestingFCDataset(data_root=args.data_root, task_root=args.task_root, task_name=args.task_name)
    if args.prediction_length is None:
        task = ds.load_task_subject(ds.subject_ids[0])
        if task.ndim == 1:
            task = task.reshape(-1, 1)
        args.prediction_length = int(task.shape[0])

    train_ds = FMRIWindowDataset(ds, args.lookback_length, args.prediction_length, args.stride, args.normalize, "train", args.train_ratio, args.val_ratio, use_task_target=True)
    val_ds = FMRIWindowDataset(ds, args.lookback_length, args.prediction_length, args.stride, args.normalize, "val", args.train_ratio, args.val_ratio, use_task_target=True)
    test_ds = FMRIWindowDataset(ds, args.lookback_length, args.prediction_length, args.stride, args.normalize, "test", args.train_ratio, args.val_ratio, use_task_target=True)
    if args.normalize and train_ds.rest_means is not None:
        val_ds.rest_means, val_ds.rest_stds = train_ds.rest_means, train_ds.rest_stds
        val_ds.task_means, val_ds.task_stds = train_ds.task_means, train_ds.task_stds
        test_ds.rest_means, test_ds.rest_stds = train_ds.rest_means, train_ds.rest_stds
        test_ds.task_means, test_ds.task_stds = train_ds.task_means, train_ds.task_stds

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    sample = next(iter(train_loader))
    V = int(sample["input"].shape[-1])

    best_path = os.path.join(args.save_dir, "best_ddpm.pth")
    if args.eval_only:
        if not os.path.isfile(best_path):
            raise FileNotFoundError(f"eval_only: checkpoint not found at {best_path}")
        ckpt = torch.load(best_path, map_location=args.device)
        # Use architecture args from checkpoint so model shape matches
        saved = ckpt.get("args") or {}
        for key in ("num_timesteps", "rest_hidden", "ctx_dim", "beta_start", "beta_end"):
            if key in saved:
                setattr(args, key, saved[key])
                print(f"[eval_only] Using {key}={saved[key]} from checkpoint", flush=True)

    model = DDPM(v_dim=V, num_timesteps=args.num_timesteps, rest_hidden=args.rest_hidden, ctx_dim=args.ctx_dim,
                 beta_start=args.beta_start, beta_end=args.beta_end).to(args.device)

    if args.eval_only:
        model.load_state_dict(ckpt["model"])
        test_metrics, subj_pred, subj_tgt, subj_starts, subj_total_len, per_subj = evaluate_subject_level_dedup(
            model, test_loader, args.device, args.prediction_length, args.sample_steps, return_for_viz=True
        )
        cfid_val = _maybe_compute_cfid(args.save_dir, args)
        print("=" * 60)
        print("DDPM TEST (eval_only, Subject-level dedup)")
        print(f"MSE: {test_metrics['mse']:.6f} ± {test_metrics['mse_std']:.6f}")
        print(f"MAE: {test_metrics['mae']:.6f} ± {test_metrics['mae_std']:.6f}")
        print(f"PSD: {test_metrics['freq_diff']:.6f} ± {test_metrics['freq_diff_std']:.6f}")
        print(f"FC sim: {test_metrics['fc_similarity']:.6f} ± {test_metrics['fc_similarity_std']:.6f}")
        if cfid_val is not None:
            print(f"cFID-FC: {cfid_val:.6f}")
        _print_topk(test_metrics)
        print("=" * 60)
        save_closest_subject_visualizations(
            per_subj, subj_pred, subj_tgt, subj_starts, subj_total_len,
            aggregate_subject_timeline,
            out_dir=args.save_dir,
            model_name="DDPM",
            fs=0.72,
        )
        test_results_path = os.path.join(args.save_dir, "test_results.txt")
        with open(test_results_path, "w", encoding="utf-8") as f:
            f.write("DDPM TEST (Subject-level dedup) [BEST CKPT]\n")
            f.write("=" * 60 + "\n")
            f.write(f"MSE (mean ± std): {test_metrics['mse']:.6f} ± {test_metrics['mse_std']:.6f}\n")
            f.write(f"MAE (mean ± std): {test_metrics['mae']:.6f} ± {test_metrics['mae_std']:.6f}\n")
            f.write(f"PSD (absolute power spectrum difference, mean ± std): {test_metrics['freq_diff']:.6f} ± {test_metrics['freq_diff_std']:.6f}\n")
            f.write(f"FC sim (mean ± std): {test_metrics['fc_similarity']:.6f} ± {test_metrics['fc_similarity_std']:.6f}\n")
            if cfid_val is not None:
                f.write(f"cFID-FC: {cfid_val:.6f}\n")
            _write_topk(f, test_metrics)
            f.write(f"Num subjects: {test_metrics['num_subjects']}\n")
            f.write("=" * 60 + "\n")
        print(f"Visualizations and test_results written to {args.save_dir}")
        return

    opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    best_val = float("inf")
    for ep in range(1, args.epochs + 1):
        tr_loss = train_epoch(
            model, train_loader, opt, args.device, args.max_grad_norm,
            pred_len=args.prediction_length,
            freq_loss_weight=args.freq_loss_weight,
            freq_aux_steps=args.freq_aux_steps,
        )
        val_metrics = evaluate_subject_level_dedup(model, val_loader, args.device, args.prediction_length, args.sample_steps)
        sched.step()
        print(f"Epoch {ep}/{args.epochs}  train_loss={tr_loss:.6f}  val_mse={val_metrics['mse']:.6f}  val_fc={val_metrics['fc_similarity']:.4f}")
        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)

    ckpt = torch.load(best_path, map_location=args.device)
    model.load_state_dict(ckpt["model"])
    test_metrics, subj_pred, subj_tgt, subj_starts, subj_total_len, per_subj = evaluate_subject_level_dedup(
        model, test_loader, args.device, args.prediction_length, args.sample_steps, return_for_viz=True
    )
    cfid_val = _maybe_compute_cfid(args.save_dir, args)
    print("=" * 60)
    print("DDPM TEST (Subject-level dedup)")
    print(f"MSE: {test_metrics['mse']:.6f} ± {test_metrics['mse_std']:.6f}")
    print(f"MAE: {test_metrics['mae']:.6f} ± {test_metrics['mae_std']:.6f}")
    print(f"PSD (absolute power spectrum difference): {test_metrics['freq_diff']:.6f} ± {test_metrics['freq_diff_std']:.6f}")
    print(f"FC sim: {test_metrics['fc_similarity']:.6f} ± {test_metrics['fc_similarity_std']:.6f}")
    if cfid_val is not None:
        print(f"cFID-FC: {cfid_val:.6f}")
    _print_topk(test_metrics)
    print("=" * 60)

    save_closest_subject_visualizations(
        per_subj, subj_pred, subj_tgt, subj_starts, subj_total_len,
        aggregate_subject_timeline,
        out_dir=args.save_dir,
        model_name="DDPM",
        fs=0.72,
    )

    test_results_path = os.path.join(args.save_dir, "test_results.txt")
    with open(test_results_path, "w", encoding="utf-8") as f:
        f.write("DDPM TEST (Subject-level dedup) [BEST CKPT]\n")
        f.write("=" * 60 + "\n")
        f.write(f"MSE (mean ± std): {test_metrics['mse']:.6f} ± {test_metrics['mse_std']:.6f}\n")
        f.write(f"MAE (mean ± std): {test_metrics['mae']:.6f} ± {test_metrics['mae_std']:.6f}\n")
        f.write(f"PSD (absolute power spectrum difference, mean ± std): {test_metrics['freq_diff']:.6f} ± {test_metrics['freq_diff_std']:.6f}\n")
        f.write(f"FC sim (mean ± std): {test_metrics['fc_similarity']:.6f} ± {test_metrics['fc_similarity_std']:.6f}\n")
        if cfid_val is not None:
            f.write(f"cFID-FC: {cfid_val:.6f}\n")
        _write_topk(f, test_metrics)
        f.write(f"Num subjects: {test_metrics['num_subjects']}\n")
        f.write("=" * 60 + "\n")
    print(f"Test results written to {test_results_path}")


if __name__ == "__main__":
    main()
