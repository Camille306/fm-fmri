#!/usr/bin/env python3
"""
TimeVAE baseline for Rest-to-Task fMRI prediction (UNCONDITIONAL DECODER).

This script trains a vanilla TimeVAE:
- Encoder: encodes REST window -> latent distribution (mu, logvar)
- Decoder: generates TASK window from z (plus time embeddings)

Important:
- This is *not* a conditional VAE; the decoder does NOT see rest directly.
- To reduce posterior collapse anyway, we implement:
  (1) Free-bits KL (per-dimension KL floor)
  (2) Slow beta warm-up (KL annealing)
  (3) No teacher forcing (prevents decoder from ignoring z)

Dataset assumptions:
- You have dataset.py providing HCPRestingFCDataset with:
    - subject_ids
    - task_root attr (or None)
    - get_subject_path(subject_id)
    - load_subject(subject_id) -> np.ndarray [T_rest, V]
    - load_task_subject(subject_id) -> np.ndarray [T_task, V]
- Rest and task have same number of ROIs (V)

Outputs:
- best_model.pth and test_results.txt in save_dir
"""

import os
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from scipy import signal
from scipy.stats import pearsonr

# ---------------------------------------------------------------------
# Import your dataset class from ../dataset.py (same pattern as yours)
# ---------------------------------------------------------------------
import importlib.util
spec = importlib.util.spec_from_file_location("dataset", str(Path(__file__).parent.parent / "dataset.py"))
dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_module)
HCPRestingFCDataset = dataset_module.HCPRestingFCDataset


# =====================================================================
# Windowed dataset: Rest window -> Task window (paired by same start idx)
# =====================================================================
class FMRIWindowDataset(Dataset):
    def __init__(
        self,
        dataset: HCPRestingFCDataset,
        lookback_length: int = 512,
        prediction_length: int = 166,
        stride: int = 100,
        normalize: bool = True,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        max_samples_per_subject: Optional[int] = None,
        norm_sample_size: int = 1000,
        norm_batch_size: int = 64,
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

        for subject_id in tqdm(subject_ids, desc=f"Indexing {self.split} subjects"):
            try:
                rest = self.dataset.load_subject(subject_id)
                task = self.dataset.load_task_subject(subject_id)
                if rest.ndim == 1:
                    rest = rest.reshape(-1, 1)
                if task.ndim == 1:
                    task = task.reshape(-1, 1)
                if rest.shape[1] != task.shape[1]:
                    continue

                Tr, V = rest.shape
                Tt, Vt = task.shape

                max_rest_idx = Tr - self.lookback_length
                max_task_idx = Tt - self.prediction_length
                max_windows = min(max_rest_idx, max_task_idx)

                count = 0
                for rest_start in range(0, max_windows + 1, self.stride):
                    task_start = rest_start
                    if task_start + self.prediction_length > Tt:
                        break
                    self.window_metadata.append(
                        {"subject_id": subject_id, "rest_start": rest_start, "task_start": task_start}
                    )
                    count += 1
                    if self.max_samples_per_subject and count >= self.max_samples_per_subject:
                        break
            except Exception:
                continue

    def _compute_normalization_stats(self, sample_size: int = 1000, batch_size: int = 64):
        if len(self.window_metadata) == 0:
            return

        idxs = np.random.choice(len(self.window_metadata), min(sample_size, len(self.window_metadata)), replace=False)

        rest_sum = rest_sum_sq = None
        task_sum = task_sum_sq = None
        rest_count = task_count = 0
        V = None

        for s in tqdm(range(0, len(idxs), batch_size), desc="Computing norm stats"):
            batch = idxs[s : s + batch_size]
            rest_list, task_list = [], []

            for i in batch:
                meta = self.window_metadata[i]
                try:
                    rest = self.dataset.load_subject(meta["subject_id"])
                    task = self.dataset.load_task_subject(meta["subject_id"])
                    if rest.ndim == 1:
                        rest = rest.reshape(-1, 1)
                    if task.ndim == 1:
                        task = task.reshape(-1, 1)
                    if V is None:
                        V = rest.shape[1]

                    r0 = meta["rest_start"]
                    t0 = meta["task_start"]
                    r = rest[r0 : r0 + self.lookback_length].astype(np.float32)
                    y = task[t0 : t0 + self.prediction_length].astype(np.float32)
                    rest_list.append(r)
                    task_list.append(y)
                except Exception:
                    continue

            if len(rest_list) == 0:
                continue

            rest_arr = np.stack(rest_list).reshape(-1, V)
            task_arr = np.stack(task_list).reshape(-1, V)

            if rest_sum is None:
                rest_sum = rest_arr.sum(axis=0)
                rest_sum_sq = (rest_arr**2).sum(axis=0)
            else:
                rest_sum += rest_arr.sum(axis=0)
                rest_sum_sq += (rest_arr**2).sum(axis=0)
            rest_count += rest_arr.shape[0]

            if task_sum is None:
                task_sum = task_arr.sum(axis=0)
                task_sum_sq = (task_arr**2).sum(axis=0)
            else:
                task_sum += task_arr.sum(axis=0)
                task_sum_sq += (task_arr**2).sum(axis=0)
            task_count += task_arr.shape[0]

        self.rest_means = rest_sum / max(rest_count, 1)
        rest_var = rest_sum_sq / max(rest_count, 1) - self.rest_means**2
        self.rest_stds = np.sqrt(np.maximum(rest_var, 0))
        self.rest_stds[self.rest_stds < 1e-8] = 1.0

        self.task_means = task_sum / max(task_count, 1)
        task_var = task_sum_sq / max(task_count, 1) - self.task_means**2
        self.task_stds = np.sqrt(np.maximum(task_var, 0))
        self.task_stds[self.task_stds < 1e-8] = 1.0

    def __len__(self):
        return len(self.window_metadata)

    def __getitem__(self, idx):
        meta = self.window_metadata[idx]
        rest = self.dataset.load_subject(meta["subject_id"])
        task = self.dataset.load_task_subject(meta["subject_id"])
        if rest.ndim == 1:
            rest = rest.reshape(-1, 1)
        if task.ndim == 1:
            task = task.reshape(-1, 1)

        r0 = meta["rest_start"]
        t0 = meta["task_start"]
        x = rest[r0 : r0 + self.lookback_length].astype(np.float32)
        y = task[t0 : t0 + self.prediction_length].astype(np.float32)

        if self.normalize and self.rest_means is not None:
            x = (x - self.rest_means) / self.rest_stds
            y = (y - self.task_means) / self.task_stds

        return {
            "input": torch.from_numpy(x),     # (L, V)
            "target": torch.from_numpy(y),    # (T, V)
            "subject_id": meta["subject_id"],
        }


# =====================================================================
# Metrics (same style as yours)
# =====================================================================
def compute_functional_connectivity(data: np.ndarray) -> np.ndarray:
    fc = np.corrcoef(data.T)
    return np.nan_to_num(fc, nan=0.0, posinf=1.0, neginf=-1.0)


def compute_frequency_difference(pred: np.ndarray, target: np.ndarray, fs: float = 0.72) -> float:
    # pred/target: (N, V)
    V = pred.shape[1]
    diffs = []
    for v in range(V):
        a = pred[:, v]
        b = target[:, v]
        try:
            _, pa = signal.welch(a, fs=fs, nperseg=min(64, len(a)))
            _, pb = signal.welch(b, fs=fs, nperseg=min(64, len(b)))
            m = min(len(pa), len(pb))
            diffs.append(np.mean(np.abs(pa[:m] - pb[:m])))
        except Exception:
            continue
    return float(np.mean(diffs)) if diffs else 0.0


def compute_fc_similarity(pred: np.ndarray, target: np.ndarray) -> float:
    fc_p = compute_functional_connectivity(pred)
    fc_t = compute_functional_connectivity(target)
    mask = np.triu(np.ones_like(fc_p, dtype=bool), k=1)
    vp, vt = fc_p[mask], fc_t[mask]
    if len(vp) > 1 and np.std(vp) > 1e-10 and np.std(vt) > 1e-10:
        r, _ = pearsonr(vp, vt)
        return float(r) if not np.isnan(r) else 0.0
    return 0.0


# =====================================================================
# TimeVAE model (vanilla)
# =====================================================================
class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, latent_dim: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, num_layers: int, output_dim: int, max_T: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.max_T = max_T

        self.z_to_h = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.z_to_c = nn.Linear(latent_dim, hidden_dim * num_layers)

        self.lstm = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.time_embed = nn.Parameter(torch.zeros(max_T, output_dim))
        nn.init.normal_(self.time_embed, mean=0.0, std=0.01)

    def forward(self, z, T: int):
        B = z.shape[0]
        if T > self.max_T:
            raise ValueError(f"T={T} exceeds max_T={self.max_T}")

        h0 = self.z_to_h(z).view(self.num_layers, B, self.hidden_dim)
        c0 = self.z_to_c(z).view(self.num_layers, B, self.hidden_dim)

        # input is purely time embeddings (no teacher forcing)
        inp = self.time_embed[:T].unsqueeze(0).repeat(B, 1, 1)  # (B, T, V)
        out, _ = self.lstm(inp, (h0, c0))
        y = self.fc(out)  # (B, T, V)
        return y


class TimeVAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, latent_dim: int, max_T: int, dropout: float):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, num_layers, latent_dim, dropout)
        self.decoder = Decoder(latent_dim, hidden_dim, num_layers, input_dim, max_T, dropout)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, T: int):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        y = self.decoder(z, T)
        return y, mu, logvar


# =====================================================================
# Loss with Free Bits (prevents posterior collapse)
# =====================================================================
def vae_loss_freebits(pred, target, mu, logvar, beta: float, free_bits: float):
    recon = F.mse_loss(pred, target, reduction="mean")

    # KL per-dim per-sample (nats): 0.5*(exp(logvar)+mu^2-1-logvar)
    kl_per_dim = 0.5 * (logvar.exp() + mu.pow(2) - 1.0 - logvar)  # (B, D)

    # Free bits: each dim contributes at least free_bits
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    kl = kl_per_dim.sum(dim=1).mean()

    total = recon + beta * kl
    return total, recon, kl


# =====================================================================
# Train / Eval
# =====================================================================
def train_one_epoch(model, loader, opt, device, T: int, beta: float, free_bits: float, max_grad_norm: float):
    model.train()
    loss_sum = recon_sum = kl_sum = 0.0
    n = 0

    for batch in tqdm(loader, desc="Training"):
        x = batch["input"].to(device).float()    # (B, L, V)
        y = batch["target"].to(device).float()   # (B, T, V)

        opt.zero_grad()
        pred, mu, logvar = model(x, T=T)
        loss, recon, kl = vae_loss_freebits(pred, y, mu, logvar, beta=beta, free_bits=free_bits)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()

        loss_sum += loss.item()
        recon_sum += recon.item()
        kl_sum += kl.item()
        n += 1

    return {"loss": loss_sum / n, "recon": recon_sum / n, "kl": kl_sum / n}


@torch.no_grad()
def evaluate(model, loader, device, T: int, compute_freq=True, compute_fc=True):
    model.eval()
    mse_sum = mae_sum = kl_sum = 0.0
    n = 0

    preds_all, tgts_all = [], []

    for batch in tqdm(loader, desc="Evaluating"):
        x = batch["input"].to(device).float()
        y = batch["target"].to(device).float()

        pred, mu, logvar = model(x, T=T)

        mse = F.mse_loss(pred, y).item()
        mae = torch.mean(torch.abs(pred - y)).item()

        kl = 0.5 * (logvar.exp() + mu.pow(2) - 1.0 - logvar)  # (B, D)
        kl = kl.sum(dim=1).mean().item()

        mse_sum += mse
        mae_sum += mae
        kl_sum += kl
        n += 1

        # flatten within batch for PSD/FC analysis
        preds_all.append(pred.reshape(-1, pred.shape[-1]).cpu().numpy())
        tgts_all.append(y.reshape(-1, y.shape[-1]).cpu().numpy())

    metrics = {"mse": mse_sum / n, "mae": mae_sum / n, "kl_loss": kl_sum / n}

    if compute_freq or compute_fc:
        P = np.concatenate(preds_all, axis=0)
        Y = np.concatenate(tgts_all, axis=0)
        if compute_freq:
            metrics["freq_diff"] = compute_frequency_difference(P, Y)
        if compute_fc:
            metrics["fc_similarity"] = compute_fc_similarity(P, Y)

    return metrics


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                       default='./data/hcp-resting-fc',
                       help='Root directory containing subject folders')
    parser.add_argument('--task_root', type=str, 
                       default='./data/hcp-task-ts',
                       help='Root directory for task data')
    parser.add_argument("--task_name", type=str, default="emotion")

    parser.add_argument("--lookback_length", type=int, default=512)
    parser.add_argument("--prediction_length", type=int, default=166)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--normalize", action="store_true", default=True)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--beta", type=float, default=0.5, help="Final beta for KL term")
    parser.add_argument("--beta_warmup_epochs", type=int, default=30, help="Slow KL warmup")
    parser.add_argument("--free_bits", type=float, default=0.05, help="KL free bits per dim (nats)")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_timevae_vanilla")
    parser.add_argument("--max_samples_per_subject", type=int, default=None)
    parser.add_argument("--norm_sample_size", type=int, default=1000)
    parser.add_argument("--norm_batch_size", type=int, default=64)

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Device: {device}")
    print(f"Save dir: {args.save_dir}")

    # dataset
    hcp = HCPRestingFCDataset(
        data_root=args.data_root,
        task_root=args.task_root,
        task_name=args.task_name,
    )
    print(f"Subjects: {len(hcp)}")

    train_ds = FMRIWindowDataset(
        dataset=hcp,
        lookback_length=args.lookback_length,
        prediction_length=args.prediction_length,
        stride=args.stride,
        normalize=args.normalize,
        split="train",
        max_samples_per_subject=args.max_samples_per_subject,
        norm_sample_size=args.norm_sample_size,
        norm_batch_size=args.norm_batch_size,
    )
    val_ds = FMRIWindowDataset(
        dataset=hcp,
        lookback_length=args.lookback_length,
        prediction_length=args.prediction_length,
        stride=args.stride,
        normalize=args.normalize,
        split="val",
        max_samples_per_subject=args.max_samples_per_subject,
        norm_sample_size=args.norm_sample_size,
        norm_batch_size=args.norm_batch_size,
    )
    test_ds = FMRIWindowDataset(
        dataset=hcp,
        lookback_length=args.lookback_length,
        prediction_length=args.prediction_length,
        stride=args.stride,
        normalize=args.normalize,
        split="test",
        max_samples_per_subject=args.max_samples_per_subject,
        norm_sample_size=args.norm_sample_size,
        norm_batch_size=args.norm_batch_size,
    )

    # share train normalization
    if args.normalize and train_ds.rest_means is not None:
        for ds in (val_ds, test_ds):
            ds.rest_means, ds.rest_stds = train_ds.rest_means, train_ds.rest_stds
            ds.task_means, ds.task_stds = train_ds.task_means, train_ds.task_stds

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device == "cuda"))

    # infer V
    sample = next(iter(train_loader))
    V = sample["input"].shape[-1]
    print(f"ROIs (V): {V}")
    T = args.prediction_length

    model = TimeVAE(
        input_dim=V,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        latent_dim=args.latent_dim,
        max_T=max(T, 256),
        dropout=args.dropout,
    ).to(device)

    opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float("inf")
    best_path = os.path.join(args.save_dir, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        # slow beta warmup
        if args.beta_warmup_epochs > 0 and epoch <= args.beta_warmup_epochs:
            beta = args.beta * (epoch / args.beta_warmup_epochs)
        else:
            beta = args.beta

        print(f"\nEpoch {epoch}/{args.epochs} | beta={beta:.4f} | free_bits={args.free_bits}")
        tr = train_one_epoch(
            model, train_loader, opt, device, T=T,
            beta=beta, free_bits=args.free_bits, max_grad_norm=args.max_grad_norm
        )
        va = evaluate(model, val_loader, device, T=T, compute_freq=False, compute_fc=False)
        sched.step()

        print(f"Train: loss={tr['loss']:.6f} recon={tr['recon']:.6f} kl={tr['kl']:.6f}")
        print(f"Val  : mse ={va['mse']:.6f} mae ={va['mae']:.6f} kl={va['kl_loss']:.6f} | lr={sched.get_last_lr()[0]:.2e}")

        if va["mse"] < best_val:
            best_val = va["mse"]
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "best_val": best_val, "args": vars(args)},
                best_path,
            )
            print(f"Saved best -> {best_path}")

    print(f"\nBest val MSE: {best_val:.6f}")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    print("\nTesting...")
    te = evaluate(model, test_loader, device, T=T, compute_freq=True, compute_fc=True)

    print("\n" + "=" * 60)
    print("Test Set Evaluation Results")
    print("=" * 60)
    print(f"MSE (Mean Squared Error): {te['mse']:.6f}")
    print(f"MAE (Mean Absolute Error): {te['mae']:.6f}")
    print(f"KL Loss: {te['kl_loss']:.6f}")
    print(f"Frequency Difference (MAE of PSD): {te.get('freq_diff', 0.0):.6f}")
    print(f"Functional Connectivity Similarity: {te.get('fc_similarity', 0.0):.6f}")
    print("=" * 60)

    out_txt = os.path.join(args.save_dir, "test_results.txt")
    with open(out_txt, "w") as f:
        for k, v in te.items():
            f.write(f"{k}: {v}\n")
    print(f"Saved test metrics -> {out_txt}")


if __name__ == "__main__":
    main()
