"""
xLSTM baseline (practical) for Rest-to-Task fMRI prediction (MSE training).

This is a dependency-free "xLSTM-inspired" baseline:
- Stacked residual LSTM blocks with LayerNorm and a small gated MLP.
- Produces a time-varying task sequence via learnable time embeddings.

Input:  rest window  (B, L, V)
Output: task sequence (B, T, V)

Checkpoint:
  - {save_dir}/best_xlstm_model.pth
  - {save_dir}/best_xlstm_model_history.csv

Usage:
  python xlstm_baseline.py --task_root /path/to/hcp-task-ts --task_name emotion
"""

import os
import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import HCPRestingFCDataset
from train import FMRIWindowDataset
from lstm_baseline import compute_frequency_difference, compute_fc_similarity


class GatedMLP(nn.Module):
    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        hidden = dim * expansion
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        u, gate = self.fc1(x).chunk(2, dim=-1)
        y = self.act(gate) * u
        y = self.drop(y)
        return self.fc2(y)


class ResidualLSTMBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0, mlp_expansion: int = 2):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.lstm = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = GatedMLP(dim, expansion=mlp_expansion, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, dim)
        h = self.ln1(x)
        h, _ = self.lstm(h)
        x = x + self.drop(h)
        h2 = self.mlp(self.ln2(x))
        x = x + self.drop(h2)
        return x


class XLSTMRestToTask(nn.Module):
    def __init__(
        self,
        num_variables: int,
        d_model: int = 128,
        n_layers: int = 3,
        dropout: float = 0.1,
        mlp_expansion: int = 2,
        max_prediction_length: int = 256,
    ):
        super().__init__()
        self.num_variables = num_variables
        self.d_model = d_model
        self.max_prediction_length = max_prediction_length

        self.in_proj = nn.Linear(num_variables, d_model)
        self.blocks = nn.ModuleList(
            [ResidualLSTMBlock(d_model, dropout=dropout, mlp_expansion=mlp_expansion) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

        self.time_embed = nn.Parameter(torch.zeros(max_prediction_length, d_model))
        nn.init.normal_(self.time_embed, mean=0.0, std=0.02)

        self.out_proj = nn.Linear(d_model, num_variables)

    def forward(self, x: torch.Tensor, prediction_length: int = 1) -> torch.Tensor:
        if prediction_length > self.max_prediction_length:
            raise ValueError(
                f"prediction_length={prediction_length} exceeds max_prediction_length={self.max_prediction_length}"
            )
        h = self.in_proj(x)  # (B, L, d_model)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        last = h[:, -1, :]  # (B, d_model)
        if prediction_length == 1:
            return self.out_proj(last)
        seq = last.unsqueeze(1) + self.time_embed[:prediction_length].unsqueeze(0)
        return self.out_proj(seq)


def train_one_epoch(model, loader, optimizer, criterion, device, prediction_length: int):
    model.train()
    total = 0.0
    n = 0
    for batch in tqdm(loader, desc="Training"):
        x = batch["input"].to(device).float()
        y = batch["target"].to(device).float()
        optimizer.zero_grad()
        pred = model(x, prediction_length=prediction_length)

        if prediction_length == 1:
            loss = criterion(pred, y)
        else:
            if y.dim() == 2:
                y = y.unsqueeze(1).repeat(1, prediction_length, 1)
            loss = criterion(pred, y)

        loss.backward()
        optimizer.step()
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


def evaluate(model, loader, device, prediction_length: int):
    model.eval()
    mse_loss = nn.MSELoss()
    total_mse = 0.0
    total_mae = 0.0
    n = 0
    all_preds = []
    all_tgts = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            x = batch["input"].to(device).float()
            y = batch["target"].to(device).float()
            pred = model(x, prediction_length=prediction_length)

            if prediction_length == 1:
                mse = mse_loss(pred, y)
                mae = torch.mean(torch.abs(pred - y))
            else:
                if y.dim() == 2:
                    y = y.unsqueeze(1).repeat(1, prediction_length, 1)
                mse = mse_loss(pred, y)
                mae = torch.mean(torch.abs(pred - y))

            total_mse += float(mse.item())
            total_mae += float(mae.item())
            n += 1

            if pred.dim() == 3:
                pred_flat = pred.reshape(-1, pred.shape[-1])
                y_flat = y.reshape(-1, y.shape[-1])
            else:
                pred_flat = pred
                y_flat = y
            all_preds.append(pred_flat.detach().cpu().numpy())
            all_tgts.append(y_flat.detach().cpu().numpy())

    preds_np = np.concatenate(all_preds, axis=0) if all_preds else np.zeros((0, 0))
    tgts_np = np.concatenate(all_tgts, axis=0) if all_tgts else np.zeros((0, 0))

    metrics = {"mse": total_mse / max(n, 1), "mae": total_mae / max(n, 1)}
    if preds_np.size and tgts_np.size:
        metrics["freq_diff"] = float(compute_frequency_difference(preds_np, tgts_np))
        metrics["fc_similarity"] = float(compute_fc_similarity(preds_np, tgts_np))
    else:
        metrics["freq_diff"] = float("nan")
        metrics["fc_similarity"] = float("nan")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train an xLSTM-inspired Rest→Task baseline (MSE).")

    # Data
    parser.add_argument("--data_root", type=str, default="./data/hcp-resting-fc")
    parser.add_argument("--task_root", type=str, default="./data/hcp-task-ts")
    parser.add_argument("--task_name", type=str, default="emotion")
    parser.add_argument("--lookback_length", type=int, default=512)
    parser.add_argument("--prediction_length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--max_samples_per_subject", type=int, default=None)
    parser.add_argument("--norm_sample_size", type=int, default=1000)
    parser.add_argument("--norm_batch_size", type=int, default=100)

    # Model
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mlp_expansion", type=int, default=2)
    parser.add_argument("--max_prediction_length", type=int, default=256)

    # Train
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_xlstm")

    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")
    os.makedirs(args.save_dir, exist_ok=True)

    ds = HCPRestingFCDataset(data_root=args.data_root, task_root=args.task_root, task_name=args.task_name)
    print(f"Found {len(ds)} subjects")

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
        max_samples_per_subject=args.max_samples_per_subject,
        use_task_target=True,
        norm_sample_size=args.norm_sample_size,
        norm_batch_size=args.norm_batch_size,
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
        use_task_target=True,
        norm_sample_size=args.norm_sample_size,
        norm_batch_size=args.norm_batch_size,
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
        use_task_target=True,
        norm_sample_size=args.norm_sample_size,
        norm_batch_size=args.norm_batch_size,
    )

    if args.normalize:
        if train_ds.rest_means is not None:
            val_ds.rest_means = train_ds.rest_means
            val_ds.rest_stds = train_ds.rest_stds
            test_ds.rest_means = train_ds.rest_means
            test_ds.rest_stds = train_ds.rest_stds
        if train_ds.task_means is not None:
            val_ds.task_means = train_ds.task_means
            val_ds.task_stds = train_ds.task_stds
            test_ds.task_means = train_ds.task_means
            test_ds.task_stds = train_ds.task_stds

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    sample = next(iter(train_loader))
    v = int(sample["input"].shape[2])
    print(f"num_variables={v}")

    model = XLSTMRestToTask(
        num_variables=v,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        mlp_expansion=args.mlp_expansion,
        max_prediction_length=max(args.max_prediction_length, args.prediction_length),
    ).to(args.device)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}\n" + "-" * 50)
        tr = train_one_epoch(model, train_loader, optimizer, criterion, args.device, args.prediction_length)
        val_metrics = evaluate(model, val_loader, args.device, args.prediction_length)
        scheduler.step()
        lr = float(scheduler.get_last_lr()[0])

        print(f"Train MSE: {tr:.6f}")
        print(f"Val MSE: {val_metrics['mse']:.6f}  Val MAE: {val_metrics['mae']:.6f}  LR: {lr:.3e}")

        history.append({"epoch": epoch, "train_mse": tr, "val_mse": val_metrics["mse"], "val_mae": val_metrics["mae"], "lr": lr})

        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            ckpt_path = os.path.join(args.save_dir, "best_xlstm_model.pth")
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_mse": best_val, "num_variables": v, "args": vars(args)},
                ckpt_path,
            )
            print(f"Saved best checkpoint to {ckpt_path}")

    hist_path = os.path.join(args.save_dir, "best_xlstm_model_history.csv")
    with open(hist_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    print(f"Saved history to {hist_path}")

    ckpt = torch.load(os.path.join(args.save_dir, "best_xlstm_model.pth"), map_location=args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    print("\nEvaluating best model on test set...")
    test_metrics = evaluate(model, test_loader, args.device, args.prediction_length)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()

