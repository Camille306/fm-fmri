"""
Train LSTM baseline with both MSE and Frequency loss, then visualize results.

This script:
1. Trains model with MSE loss → saves as 'best_model_mse.pth'
2. Trains model with Frequency loss → saves as 'best_model_freq.pth'
3. Trains model with FC loss → saves as 'best_model_fc.pth'
3. Evaluates both models and finds closest matching FC subject
4. Creates visualizations for each model:
   - Task FC comparison (real vs generated)
   - Frequency/power spectrum comparison
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy import signal
from scipy.stats import pearsonr
from tqdm import tqdm
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# seaborn is optional; we only use matplotlib for plots

# Import from lstm_baseline
from dataset import HCPRestingFCDataset
from train import FMRIWindowDataset
from lstm_baseline import (
    LSTMBaseline, 
    FrequencyLoss,
    compute_functional_connectivity,
    compute_fc_similarity,
    train_epoch,
    evaluate
)

class FunctionalConnectivityLoss(nn.Module):
    """
    Differentiable FC loss.

    Computes correlation matrices across the time dimension and penalizes the
    difference between predicted and target FC.

    Expected input:
    - pred:   (B, T, V)
    - target: (B, T, V)
    """

    def __init__(self, eps: float = 1e-6, use_upper_triangle: bool = True, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        self.use_upper_triangle = use_upper_triangle
        self.reduction = reduction

    def _batch_corrcoef(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, V)
        x = x - x.mean(dim=1, keepdim=True)
        b, t, v = x.shape
        if t < 2:
            # Degenerate case: cannot compute correlation from <2 points
            return torch.zeros((b, v, v), device=x.device, dtype=x.dtype)

        cov = torch.einsum("btv,btw->bvw", x, x) / (t - 1)
        var = torch.diagonal(cov, dim1=-2, dim2=-1)  # (B, V)
        std = torch.sqrt(torch.clamp(var, min=self.eps))
        denom = std.unsqueeze(2) * std.unsqueeze(1)  # (B, V, V)
        corr = cov / (denom + self.eps)
        return torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # If single step, FC is undefined → fall back to MSE
        if pred.dim() == 2:
            return nn.functional.mse_loss(pred, target)

        if pred.dim() != 3:
            raise ValueError(f"Expected pred dim 2 or 3, got {pred.shape}")

        if target.dim() == 2:
            target = target.unsqueeze(1).repeat(1, pred.shape[1], 1)

        b, t, v = pred.shape
        if t < 2:
            return nn.functional.mse_loss(pred, target)

        fc_pred = self._batch_corrcoef(pred)    # (B, V, V)
        fc_tgt = self._batch_corrcoef(target)   # (B, V, V)

        if self.use_upper_triangle:
            mask = torch.triu(torch.ones((v, v), device=pred.device, dtype=torch.bool), diagonal=1)
            a = fc_pred[:, mask]
            b_ = fc_tgt[:, mask]
        else:
            a = fc_pred.reshape(b, -1)
            b_ = fc_tgt.reshape(b, -1)

        loss = (a - b_) ** 2
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        raise ValueError(f"Unknown reduction: {self.reduction}")

class FCSimilarityLoss(nn.Module):
    """
    Differentiable version of compute_fc_similarity():
    - build FC matrices (corr across time)
    - take upper triangle (excluding diagonal)
    - compute Pearson correlation between FC vectors
    - loss = 1 - corr (maximize similarity)
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def _batch_corrcoef(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, V)
        x = x - x.mean(dim=1, keepdim=True)
        b, t, v = x.shape
        if t < 2:
            return torch.zeros((b, v, v), device=x.device, dtype=x.dtype)
        cov = torch.einsum("btv,btw->bvw", x, x) / (t - 1)
        var = torch.diagonal(cov, dim1=-2, dim2=-1)  # (B, V)
        std = torch.sqrt(torch.clamp(var, min=self.eps))
        corr = cov / (std.unsqueeze(2) * std.unsqueeze(1) + self.eps)
        return torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # If single step, similarity is undefined → fall back to MSE
        if pred.dim() == 2:
            return nn.functional.mse_loss(pred, target)
        if pred.dim() != 3:
            raise ValueError(f"Expected pred dim 2 or 3, got {pred.shape}")
        if target.dim() == 2:
            target = target.unsqueeze(1).repeat(1, pred.shape[1], 1)

        b, t, v = pred.shape
        if t < 2:
            return nn.functional.mse_loss(pred, target)

        fc_pred = self._batch_corrcoef(pred)   # (B, V, V)
        fc_tgt = self._batch_corrcoef(target)  # (B, V, V)

        mask = torch.triu(torch.ones((v, v), device=pred.device, dtype=torch.bool), diagonal=1)
        a = fc_pred[:, mask]  # (B, M)
        b_ = fc_tgt[:, mask]  # (B, M)

        a = a - a.mean(dim=1, keepdim=True)
        b_ = b_ - b_.mean(dim=1, keepdim=True)
        a_std = torch.sqrt(torch.clamp((a * a).mean(dim=1, keepdim=True), min=self.eps))
        b_std = torch.sqrt(torch.clamp((b_ * b_).mean(dim=1, keepdim=True), min=self.eps))
        corr = (a * b_).mean(dim=1, keepdim=False) / (a_std.squeeze(1) * b_std.squeeze(1) + self.eps)  # (B,)

        loss = 1.0 - corr  # want corr -> 1
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        raise ValueError(f"Unknown reduction: {self.reduction}")

def _write_csv(rows: list[dict], out_path: str) -> None:
    if not rows:
        return
    # Keep a stable column order
    cols = list(rows[0].keys())
    import csv
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _plot_training_curves(rows: list[dict], title: str, out_path: str) -> None:
    if not rows:
        return
    epochs = [r["epoch"] for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    val_mse = [r["val_mse"] for r in rows]
    val_mae = [r["val_mae"] for r in rows]
    lr = [r["lr"] for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax0, ax1 = axes

    ax0.plot(epochs, train_loss, label="train_loss", linewidth=2)
    ax0.plot(epochs, val_mse, label="val_mse", linewidth=2)
    ax0.set_ylabel("Loss")
    ax0.grid(True, alpha=0.3)
    ax0.legend()

    ax1.plot(epochs, val_mae, label="val_mae", linewidth=2, color="tab:orange")
    ax1.set_ylabel("MAE")
    ax1.set_xlabel("Epoch")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # LR on secondary axis
    ax1b = ax1.twinx()
    ax1b.plot(epochs, lr, label="lr", linewidth=1.5, color="tab:green", alpha=0.8)
    ax1b.set_ylabel("LR")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _np_stats(x: np.ndarray) -> str:
    x = np.asarray(x)
    if x.size == 0:
        return "empty"
    finite = np.isfinite(x)
    if not finite.any():
        return f"shape={x.shape} all_non_finite"
    xf = x[finite]
    frac_zero = float(np.mean(x == 0.0))
    return (
        f"shape={x.shape} dtype={x.dtype} "
        f"min={xf.min():.6g} max={xf.max():.6g} mean={xf.mean():.6g} std={xf.std():.6g} "
        f"finite={finite.mean():.2%} frac_zero={frac_zero:.2%}"
    )

def _maybe_denorm(arr: np.ndarray, mean: np.ndarray | None, std: np.ndarray | None) -> np.ndarray:
    if mean is None or std is None:
        return arr
    # mean/std are (V,), arr is (T,V)
    return arr * std.reshape(1, -1) + mean.reshape(1, -1)


def train_model_with_loss(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    prediction_length,
    save_dir,
    checkpoint_name,
    dropout=0.1,
    max_prediction_length: int = 256,
):
    """
    Train a model and save the best checkpoint.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        epochs: Number of epochs
        prediction_length: Prediction length
        save_dir: Directory to save checkpoint
        checkpoint_name: Name for the checkpoint file
        
    Returns:
        Best validation loss
    """
    best_val_loss = float('inf')
    best_epoch = 0
    history_rows: list[dict] = []
    
    print(f"\n{'='*60}")
    print(f"Training with checkpoint name: {checkpoint_name}")
    print(f"{'='*60}")
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 50)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, 
            prediction_length=prediction_length
        )
        
        # Validate
        val_metrics = evaluate(
            model, val_loader, criterion, device,
            prediction_length=prediction_length,
            compute_freq_diff=False, compute_fc_sim=False
        )
        val_loss = val_metrics['mse']
        
        # Update learning rate
        scheduler.step()
        current_lr = float(scheduler.get_last_lr()[0])
        
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss (MSE): {val_loss:.6f}, Val MAE: {val_metrics['mae']:.6f}")
        print(f"LR: {current_lr:.6e}")

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_mse": float(val_loss),
                "val_mae": float(val_metrics["mae"]),
                "lr": current_lr,
            }
        )
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            checkpoint_path = os.path.join(save_dir, checkpoint_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'num_variables': model.input_dim,
                'args': {
                    'hidden_dim': model.hidden_dim,
                    'num_layers': model.num_layers,
                    'dropout': dropout,
                    'input_dim': model.input_dim,
                    'output_dim': model.output_dim,
                    'prediction_length': prediction_length,
                    'max_prediction_length': max_prediction_length,
                }
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
    
    # Export training curves + CSV for hyperparameter tuning
    stem = os.path.splitext(checkpoint_name)[0]
    csv_path = os.path.join(save_dir, f"{stem}_history.csv")
    png_path = os.path.join(save_dir, f"{stem}_history.png")
    _write_csv(history_rows, csv_path)
    _plot_training_curves(history_rows, title=f"{stem} training curves", out_path=png_path)
    print(f"Saved training history CSV: {csv_path}")
    print(f"Saved training history plot: {png_path}")

    print(f"\nTraining complete! Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
    return best_val_loss, history_rows


def predict_full_task_sequence(
    model,
    dataloader,
    device,
    prediction_length=1,
    debug: bool = False,
):
    """
    Generate predictions for full task sequences, aggregated by subject.
    
    Note: The output IS sliced into windows during training/evaluation.
    - Each window predicts 'prediction_length' time steps
    - If prediction_length = full task sequence length, typically only 1 window per subject
    - If prediction_length < full task length, multiple overlapping windows exist
    - This function aggregates window predictions back into full sequences
    
    Handles overlapping windows by taking the first window's prediction
    (when prediction_length = full task length, there's typically no overlap).
    
    Returns:
        Dictionary mapping subject_id to (predictions, targets) tuples
        predictions: (num_timepoints, num_variables) - aggregated from windows
        targets: (num_timepoints, num_variables) - aggregated from windows
    """
    model.eval()
    # Store per-window predictions/targets grouped by subject.
    # If prediction_length == full task length (e.g., 176), typically there is only 1 window per subject.
    subject_pred_dict = {}   # {subject_id: [ (prediction_length, num_vars), ... ]}
    subject_target_dict = {} # {subject_id: [ (prediction_length, num_vars), ... ]}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            input_seq = batch['input'].to(device).float()
            target = batch['target'].to(device).float()
            subject_ids = batch['subject_id']
            
            batch_size = input_seq.shape[0]
            
            # Generate predictions
            pred = model(input_seq, prediction_length=prediction_length)

            # One-time debug dump for first batch
            if debug and not hasattr(predict_full_task_sequence, "_printed_batch_debug"):
                predict_full_task_sequence._printed_batch_debug = True  # type: ignore[attr-defined]
                in_np = input_seq.detach().cpu().numpy()
                tgt_np = target.detach().cpu().numpy()
                pred_np_dbg = pred.detach().cpu().numpy()
                print("\n[DEBUG] First eval batch stats")
                print(f"  input_seq: {_np_stats(in_np)}")
                print(f"  target:    {_np_stats(tgt_np)}")
                print(f"  pred:      {_np_stats(pred_np_dbg)}")
            
            # Handle predictions
            if pred.dim() == 3:
                # (batch_size, prediction_length, num_variables)
                pred_np = pred.cpu().numpy()
            else:
                # (batch_size, num_variables) -> expand to (batch_size, 1, num_variables)
                pred_np = pred.cpu().numpy()[:, np.newaxis, :]
            
            # Handle targets
            if target.dim() == 3:
                # (batch_size, prediction_length, num_variables)
                target_np = target.cpu().numpy()
            else:
                # (batch_size, num_variables) -> expand to (batch_size, 1, num_variables)
                target_np = target.cpu().numpy()[:, np.newaxis, :]
            
            # Get window metadata from dataset to know time indices
            # Note: We need to track which time window each prediction corresponds to
            # Since we don't have direct access to window indices in the batch,
            # we'll use a simpler approach: if prediction_length == full task length,
            # there should be minimal/no overlap. Otherwise, we'll need to track indices.
            
            # For now, use a simpler aggregation: collect all predictions and targets
            # We'll handle overlap by checking if prediction_length matches expected task length
            for i in range(batch_size):
                subject_id = subject_ids[i]
                
                if subject_id not in subject_pred_dict:
                    subject_pred_dict[subject_id] = []
                    subject_target_dict[subject_id] = []
                
                # Store predictions and targets for this window
                subject_pred_dict[subject_id].append(pred_np[i])  # (prediction_length, num_variables)
                subject_target_dict[subject_id].append(target_np[i])  # (prediction_length, num_variables)
    
    # Aggregate predictions for each subject
    subject_data = {}
    for subject_id in subject_pred_dict.keys():
        pred_list = subject_pred_dict[subject_id]
        target_list = subject_target_dict[subject_id]
        
        if len(pred_list) == 0:
            print(f"  Warning: Subject {subject_id} has no predictions!")
            continue
        
        # If we have multiple windows, they might overlap
        # Simple approach: if prediction_length is the full task sequence,
        # there should be only one window (or very few). Otherwise, we concatenate
        # and take unique time points (or average overlaps)
        
        if len(pred_list) == 1:
            # Single window - no overlap to worry about
            preds = pred_list[0]  # (prediction_length, num_variables)
            targets = target_list[0]  # (prediction_length, num_variables)
        else:
            # Multiple windows - concatenate them
            preds = np.concatenate(pred_list, axis=0)  # (num_windows * prediction_length, num_variables)
            targets = np.concatenate(target_list, axis=0)  # (num_windows * prediction_length, num_variables)
            
            # If prediction_length == 1, each window predicts one time point
            # In this case, concatenating gives us the full sequence (no truncation needed)
            # If prediction_length > 1 and we have overlap, we might need to handle it differently
            # For now, if prediction_length > 1 and we have multiple windows, 
            # it means there's overlap - take the first window's full prediction
            if prediction_length > 1 and preds.shape[0] > prediction_length:
                print(f"  Info: Subject {subject_id} has {len(pred_list)} windows with overlap.")
                print(f"    Total predicted time points: {preds.shape[0]}, prediction_length: {prediction_length}")
                print(f"    Taking first {prediction_length} time points (from first window)")
                preds = preds[:prediction_length]
                targets = targets[:prediction_length]
            # If prediction_length == 1, keep all concatenated predictions (they form the full sequence)
        
        if debug:
            print(f"  Subject {subject_id}: preds shape={preds.shape}, targets shape={targets.shape}")
            print(f"    preds range: [{preds.min():.4f}, {preds.max():.4f}], mean={preds.mean():.4f}")
            print(f"    targets range: [{targets.min():.4f}, {targets.max():.4f}], mean={targets.mean():.4f}")
        
        subject_data[subject_id] = (preds, targets)
    
    if debug:
        print(f"\nTotal subjects with predictions: {len(subject_data)}")
    return subject_data


def find_closest_subject(subject_data):
    """
    Find the subject with the highest FC similarity.
    
    Returns:
        Tuple of (best_subject_id, best_fc_sim, fc_pred, fc_target, predictions, targets)
    """
    best_subject_id = None
    best_fc_sim = -1.0
    best_fc_pred = None
    best_fc_target = None
    best_predictions = None
    best_targets = None
    
    print("\nComputing FC similarity for each subject...")
    for subject_id, (predictions, targets) in tqdm(subject_data.items(), desc="Finding closest subject"):
        # Compute FC similarity
        fc_sim = compute_fc_similarity(predictions, targets)
        
        if fc_sim > best_fc_sim:
            best_fc_sim = fc_sim
            best_subject_id = subject_id
            best_fc_pred = compute_functional_connectivity(predictions)
            best_fc_target = compute_functional_connectivity(targets)
            best_predictions = predictions
            best_targets = targets
    
    print(f"\nClosest subject: {best_subject_id}")
    print(f"FC Similarity: {best_fc_sim:.4f}")
    
    return best_subject_id, best_fc_sim, best_fc_pred, best_fc_target, best_predictions, best_targets


def compute_power_spectrum(data, fs=0.72):
    """
    Compute power spectral density using Welch's method.
    
    Args:
        data: Time series data, shape (num_timepoints, num_variables)
        fs: Sampling frequency in Hz
        
    Returns:
        Tuple of (frequencies, psd) where psd is averaged across variables
    """
    num_timepoints, num_variables = data.shape
    all_freqs = []
    all_psd = []
    
    for var_idx in range(num_variables):
        signal_data = data[:, var_idx]
        if len(signal_data) < 4:
            continue
        
        freqs, psd = signal.welch(
            signal_data, 
            fs=fs, 
            nperseg=min(64, len(signal_data))
        )
        all_freqs.append(freqs)
        all_psd.append(psd)
    
    if len(all_freqs) == 0:
        return None, None
    
    # Use the first frequency array (they should all be the same length)
    freqs = all_freqs[0]
    # Average PSD across all variables
    psd_mean = np.mean(all_psd, axis=0)
    
    return freqs, psd_mean


def plot_fc_matrices(fc_pred, fc_target, subject_id, fc_similarity, save_path):
    """Plot predicted and real task FC matrices side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Determine common color scale
    vmin = min(fc_pred.min(), fc_target.min())
    vmax = max(fc_pred.max(), fc_target.max())
    
    # Plot predicted FC
    im1 = axes[0].imshow(fc_pred, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0].set_title(f'Predicted Task FC\nSubject: {subject_id}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('ROI Index', fontsize=12)
    axes[0].set_ylabel('ROI Index', fontsize=12)
    plt.colorbar(im1, ax=axes[0], label='Correlation')
    
    # Plot real FC
    im2 = axes[1].imshow(fc_target, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
    axes[1].set_title(f'Real Task FC\nSubject: {subject_id}', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('ROI Index', fontsize=12)
    axes[1].set_ylabel('ROI Index', fontsize=12)
    plt.colorbar(im2, ax=axes[1], label='Correlation')
    
    # Add overall title
    fig.suptitle(
        f'Task Functional Connectivity Comparison\nFC Similarity: {fc_similarity:.4f}',
        fontsize=16,
        fontweight='bold',
        y=1.02
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved FC plot to {save_path}")


def plot_time_series(predictions, targets, subject_id, save_path, num_rois_to_plot=5):
    """
    Plot full time series comparison between predicted and real task data.
    
    Args:
        predictions: Predicted task data, shape (num_timepoints, num_variables)
        targets: Real task data, shape (num_timepoints, num_variables)
        subject_id: Subject ID string
        save_path: Path to save the figure
        num_rois_to_plot: Number of ROIs to visualize (default: 5)
    """
    num_timepoints, num_variables = predictions.shape
    
    # Select a few representative ROIs to plot
    roi_indices = np.linspace(0, num_variables - 1, num_rois_to_plot, dtype=int)
    
    fig, axes = plt.subplots(num_rois_to_plot, 1, figsize=(14, 3 * num_rois_to_plot))
    if num_rois_to_plot == 1:
        axes = [axes]
    
    time_steps = np.arange(num_timepoints)
    
    for idx, roi_idx in enumerate(roi_indices):
        ax = axes[idx]
        
        # Plot real time series
        ax.plot(time_steps, targets[:, roi_idx], 'b-', linewidth=2, 
                label='Real Task Data', alpha=0.8)
        
        # Plot predicted time series
        ax.plot(time_steps, predictions[:, roi_idx], 'r--', linewidth=2, 
                label='Predicted Task Data', alpha=0.8)
        
        # Calculate metrics for this ROI
        mse = np.mean((predictions[:, roi_idx] - targets[:, roi_idx]) ** 2)
        mae = np.mean(np.abs(predictions[:, roi_idx] - targets[:, roi_idx]))
        corr = np.corrcoef(predictions[:, roi_idx], targets[:, roi_idx])[0, 1]
        
        ax.set_title(f'ROI {roi_idx + 1}/{num_variables} - MSE: {mse:.4f}, MAE: {mae:.4f}, Corr: {corr:.4f}',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Signal Amplitude', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(
        f'Full Time Series Prediction - Subject {subject_id}\n'
        f'Total Time Points: {num_timepoints}, Total ROIs: {num_variables}',
        fontsize=14,
        fontweight='bold',
        y=0.995
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved time series plot to {save_path}")


def plot_frequency_spectrum(predictions, targets, subject_id, save_path, fs=0.72):
    """
    Plot power spectral density comparison between predicted and real task data.
    
    Args:
        predictions: Predicted task data, shape (num_timepoints, num_variables)
        targets: Real task data, shape (num_timepoints, num_variables)
        subject_id: Subject ID string
        save_path: Path to save the figure
        fs: Sampling frequency in Hz
    """
    # Compute power spectra
    freqs_pred, psd_pred = compute_power_spectrum(predictions, fs=fs)
    freqs_target, psd_target = compute_power_spectrum(targets, fs=fs)
    
    if freqs_pred is None or freqs_target is None:
        print(f"Warning: Could not compute power spectrum for subject {subject_id}")
        return
    
    # Ensure same frequency grid
    if len(freqs_pred) != len(freqs_target):
        min_len = min(len(freqs_pred), len(freqs_target))
        freqs_pred = freqs_pred[:min_len]
        freqs_target = freqs_target[:min_len]
        psd_pred = psd_pred[:min_len]
        psd_target = psd_target[:min_len]
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Overlaid power spectra
    ax1 = axes[0]
    ax1.plot(freqs_pred, psd_pred, 'r-', linewidth=2, label='Predicted', alpha=0.8)
    ax1.plot(freqs_target, psd_target, 'b-', linewidth=2, label='Real', alpha=0.8)
    ax1.fill_between(freqs_pred, psd_pred, alpha=0.3, color='red')
    ax1.fill_between(freqs_target, psd_target, alpha=0.3, color='blue')
    ax1.set_xlabel('Frequency (Hz)', fontsize=12)
    ax1.set_ylabel('Power Spectral Density', fontsize=12)
    ax1.set_title(f'Power Spectral Density Comparison\nSubject: {subject_id}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, min(freqs_pred.max(), freqs_target.max())])
    
    # Plot 2: Power spectrum difference
    ax2 = axes[1]
    psd_diff = psd_pred - psd_target
    ax2.plot(freqs_pred, psd_diff, 'g-', linewidth=2, alpha=0.8)
    ax2.fill_between(freqs_pred, psd_diff, alpha=0.3, color='green')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Frequency (Hz)', fontsize=12)
    ax2.set_ylabel('PSD Difference (Predicted - Real)', fontsize=12)
    ax2.set_title('Power Spectral Density Difference', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, min(freqs_pred.max(), freqs_target.max())])
    
    # Compute and display metrics
    mse_psd = np.mean((psd_pred - psd_target) ** 2)
    mae_psd = np.mean(np.abs(psd_diff))
    
    fig.suptitle(
        f'Frequency Analysis - Subject {subject_id}\n'
        f'PSD MSE: {mse_psd:.6f}, PSD MAE: {mae_psd:.6f}',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved frequency spectrum plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train LSTM with MSE and Frequency loss, then visualize')
    
    # Data arguments
    parser.add_argument('--data_root', type=str,
                       default='./data/hcp-resting-fc',
                       help='Root directory containing subject folders')
    parser.add_argument('--task_root', type=str,
                       default='./data/hcp-task-ts',
                       help='Root directory for task data')
    parser.add_argument('--task_name', type=str, default='emotion',
                       help='Name of the task (e.g., "emotion")')
    parser.add_argument('--lookback_length', type=int, default=512,
                       help='Number of time steps to use as input context')
    parser.add_argument('--prediction_length', type=int, default=None,
                       help='Number of time steps to predict (None = predict full task sequence)')
    parser.add_argument('--stride', type=int, default=100,
                       help='Stride for sliding window')
    parser.add_argument('--normalize', action='store_true', default=True,
                       help='Normalize data to zero mean and unit variance')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Proportion of data for training')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Proportion of data for validation')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension of LSTM')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay for optimizer')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu). Auto-detect if not specified')
    parser.add_argument('--num_workers', type=int, default=1,
                       help='Number of data loader workers')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_dual_loss',
                       help='Directory to save checkpoints')
    parser.add_argument('--max_samples_per_subject', type=int, default=None,
                       help='Maximum windows per subject (None = all)')
    parser.add_argument('--norm_sample_size', type=int, default=1000,
                       help='Number of samples to use for computing normalization stats')
    parser.add_argument('--norm_batch_size', type=int, default=100,
                       help='Batch size for computing normalization stats')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training and only do visualization (requires existing checkpoints)')
    parser.add_argument('--debug', action='store_true',
                       help='Verbose debug prints (per-subject shapes/ranges, etc.)')
    parser.add_argument('--denormalize_plots', action='store_true', default=True,
                       help='Denormalize predictions/targets (uses train split stats) before plotting')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {args.device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    vis_dir = os.path.join(args.save_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load dataset
    print("Loading HCP resting-state fMRI dataset...")
    print(f"  Task name: {args.task_name}")
    print("  Mode: Rest-to-Task prediction")
    
    hcp_dataset = HCPRestingFCDataset(
        data_root=args.data_root,
        task_root=args.task_root,
        task_name=args.task_name
    )
    print(f"Found {len(hcp_dataset)} subjects")
    
    # Determine prediction length
    if args.prediction_length is None:
        print("Prediction length not specified, inferring from task data...")
        try:
            # Check multiple subjects to ensure consistency
            prediction_lengths = []
            num_subjects_to_check = min(10, len(hcp_dataset.subject_ids))
            
            for i in range(num_subjects_to_check):
                try:
                    subject_id = hcp_dataset.subject_ids[i]
                    task_data = hcp_dataset.load_task_subject(subject_id)
                    if len(task_data.shape) == 1:
                        task_data = task_data.reshape(-1, 1)
                    prediction_lengths.append(task_data.shape[0])
                except Exception as e:
                    continue
            
            if len(prediction_lengths) > 0:
                # Use the most common prediction length (mode)
                unique_lengths, counts = np.unique(prediction_lengths, return_counts=True)
                most_common_idx = np.argmax(counts)
                args.prediction_length = int(unique_lengths[most_common_idx])
                
                print(f"  Checked {len(prediction_lengths)} subjects")
                print(f"  Found prediction lengths: {unique_lengths} (counts: {counts})")
                print(f"  Using prediction_length: {args.prediction_length} (most common, full task sequence)")
                
                # Warn if there's inconsistency
                if len(unique_lengths) > 1:
                    print(f"  Warning: Found {len(unique_lengths)} different task lengths across subjects!")
                    print(f"  Using the most common length: {args.prediction_length}")
            else:
                raise ValueError("Could not load task data from any subject")
                
        except Exception as e:
            print(f"  Warning: Could not infer prediction length: {e}")
            print("  Using default: 1")
            args.prediction_length = 1
    
    print(f"Using stride: {args.stride}")
    print(f"Using prediction_length: {args.prediction_length}")
    
    # Create windowed datasets
    train_dataset = FMRIWindowDataset(
        dataset=hcp_dataset,
        lookback_length=args.lookback_length,
        prediction_length=args.prediction_length,
        stride=args.stride,
        normalize=args.normalize,
        split='train',
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_samples_per_subject=args.max_samples_per_subject,
        use_task_target=True,
        norm_sample_size=args.norm_sample_size,
        norm_batch_size=args.norm_batch_size
    )
    
    val_dataset = FMRIWindowDataset(
        dataset=hcp_dataset,
        lookback_length=args.lookback_length,
        prediction_length=args.prediction_length,
        stride=args.stride,
        normalize=args.normalize,
        split='val',
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_samples_per_subject=args.max_samples_per_subject,
        use_task_target=True,
        norm_sample_size=args.norm_sample_size,
        norm_batch_size=args.norm_batch_size
    )
    
    test_dataset = FMRIWindowDataset(
        dataset=hcp_dataset,
        lookback_length=args.lookback_length,
        prediction_length=args.prediction_length,
        stride=args.stride,
        normalize=args.normalize,
        split='test',
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_samples_per_subject=args.max_samples_per_subject,
        use_task_target=True,
        norm_sample_size=args.norm_sample_size,
        norm_batch_size=args.norm_batch_size
    )
    
    # Use normalization stats from training set
    if args.normalize:
        if train_dataset.rest_means is not None:
            val_dataset.rest_means = train_dataset.rest_means
            val_dataset.rest_stds = train_dataset.rest_stds
            test_dataset.rest_means = train_dataset.rest_means
            test_dataset.rest_stds = train_dataset.rest_stds
        if train_dataset.task_means is not None:
            val_dataset.task_means = train_dataset.task_means
            val_dataset.task_stds = train_dataset.task_stds
            test_dataset.task_means = train_dataset.task_means
            test_dataset.task_stds = train_dataset.task_stds
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    # Get number of variables
    sample_batch = next(iter(train_loader))
    num_variables = sample_batch['input'].shape[2]
    print(f"Number of variables (ROIs): {num_variables}")

    # Keep task denorm stats for plotting (if available)
    task_means = train_dataset.task_means if hasattr(train_dataset, "task_means") else None
    task_stds = train_dataset.task_stds if hasattr(train_dataset, "task_stds") else None
    
    # Training loop for both loss functions
    checkpoint_names = {
        'mse': 'best_model_mse.pth',
        'freq': 'best_model_freq.pth',
        # FC similarity objective (1 - compute_fc_similarity)
        'fc': 'best_model_fcsim.pth',
    }
    
    if not args.skip_training:
        # Train with MSE loss
        print("\n" + "="*60)
        print("TRAINING WITH MSE LOSS")
        print("="*60)
        
        model_mse = LSTMBaseline(
            input_dim=num_variables,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            output_dim=num_variables,
            dropout=args.dropout,
            max_prediction_length=max(256, args.prediction_length),
        ).to(args.device)
        
        criterion_mse = nn.MSELoss()
        optimizer_mse = torch.optim.Adam(
            model_mse.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )
        scheduler_mse = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_mse, T_max=args.epochs
        )
        
        _, mse_history = train_model_with_loss(
            model_mse, train_loader, val_loader, criterion_mse,
            optimizer_mse, scheduler_mse, args.device, args.epochs,
            args.prediction_length, args.save_dir, checkpoint_names['mse'],
            dropout=args.dropout,
            max_prediction_length=max(256, args.prediction_length),
        )
        
        # Train with Frequency loss
        print("\n" + "="*60)
        print("TRAINING WITH FREQUENCY LOSS")
        print("="*60)
        
        model_freq = LSTMBaseline(
            input_dim=num_variables,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            output_dim=num_variables,
            dropout=args.dropout,
            max_prediction_length=max(256, args.prediction_length),
        ).to(args.device)
        
        # Use the FFT-based FrequencyLoss (second definition in lstm_baseline.py)
        # This one works with tensors and preserves gradients
        criterion_freq = FrequencyLoss(reduction='mean')
        optimizer_freq = torch.optim.Adam(
            model_freq.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        scheduler_freq = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_freq, T_max=args.epochs
        )
        
        _, freq_history = train_model_with_loss(
            model_freq, train_loader, val_loader, criterion_freq,
            optimizer_freq, scheduler_freq, args.device, args.epochs,
            args.prediction_length, args.save_dir, checkpoint_names['freq'],
            dropout=args.dropout,
            max_prediction_length=max(256, args.prediction_length),
        )

        # Train with FC loss
        print("\n" + "="*60)
        print("TRAINING WITH FC-SIMILARITY LOSS")
        print("="*60)

        model_fc = LSTMBaseline(
            input_dim=num_variables,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            output_dim=num_variables,
            dropout=args.dropout,
            max_prediction_length=max(256, args.prediction_length),
        ).to(args.device)

        # Optimize the same metric used by find_closest_subject(): compute_fc_similarity
        # (implemented in a differentiable way as 1 - corr(FC_pred_vec, FC_target_vec)).
        criterion_fc = FCSimilarityLoss(eps=1e-6, reduction="mean")
        optimizer_fc = torch.optim.Adam(
            model_fc.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler_fc = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_fc, T_max=args.epochs
        )

        _, fc_history = train_model_with_loss(
            model_fc, train_loader, val_loader, criterion_fc,
            optimizer_fc, scheduler_fc, args.device, args.epochs,
            args.prediction_length, args.save_dir, checkpoint_names['fc'],
            dropout=args.dropout,
            max_prediction_length=max(256, args.prediction_length),
        )
    
    # Evaluation and visualization
    print("\n" + "="*60)
    print("EVALUATION AND VISUALIZATION")
    print("="*60)
    
    for loss_type, checkpoint_name in checkpoint_names.items():
        checkpoint_path = os.path.join(args.save_dir, checkpoint_name)
        
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint {checkpoint_path} not found. Skipping {loss_type} visualization.")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {loss_type.upper()} model")
        print(f"{'='*60}")
        
        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        ck_max_pred_len = checkpoint.get('args', {}).get('max_prediction_length', 256)
        model = LSTMBaseline(
            input_dim=checkpoint['num_variables'],
            hidden_dim=checkpoint['args']['hidden_dim'],
            num_layers=checkpoint['args']['num_layers'],
            output_dim=checkpoint['num_variables'],
            dropout=checkpoint['args']['dropout'],
            max_prediction_length=ck_max_pred_len,
        ).to(args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded {loss_type} model from {checkpoint_path}")
        
        # Get prediction_length from checkpoint if available, otherwise use args
        checkpoint_prediction_length = checkpoint.get('args', {}).get('prediction_length', None)
        if checkpoint_prediction_length is not None:
            eval_prediction_length = checkpoint_prediction_length
            print(f"  Using prediction_length from checkpoint: {eval_prediction_length}")
        else:
            eval_prediction_length = args.prediction_length
            print(f"  Using prediction_length from args: {eval_prediction_length}")
        
        # Generate predictions
        subject_data = predict_full_task_sequence(
            model, test_loader, args.device, eval_prediction_length, debug=args.debug
        )
        
        # Find closest subject
        if len(subject_data) == 0:
            print(f"  Error: No subject data generated for {loss_type} model!")
            continue
            
        best_subject_id, best_fc_sim, fc_pred, fc_target, predictions, targets = find_closest_subject(subject_data)

        # Optionally denormalize for plotting/PSD (FC is scale-invariant, but PSD is not)
        if args.denormalize_plots:
            predictions_plot = _maybe_denorm(predictions, task_means, task_stds)
            targets_plot = _maybe_denorm(targets, task_means, task_stds)
        else:
            predictions_plot = predictions
            targets_plot = targets
        
        if args.debug:
            print(f"\n  Best subject {best_subject_id}:")
            print(f"    Predictions shape: {predictions.shape}, range: [{predictions.min():.4f}, {predictions.max():.4f}]")
            print(f"    Targets shape: {targets.shape}, range: [{targets.min():.4f}, {targets.max():.4f}]")
            print(f"    FC pred shape: {fc_pred.shape}, FC target shape: {fc_target.shape}")
        
        # Visualize FC
        fc_plot_path = os.path.join(vis_dir, f'fc_comparison_{loss_type}_{best_subject_id}.png')
        plot_fc_matrices(fc_pred, fc_target, best_subject_id, best_fc_sim, fc_plot_path)
        
        # Visualize frequency spectrum
        freq_plot_path = os.path.join(vis_dir, f'frequency_spectrum_{loss_type}_{best_subject_id}.png')
        plot_frequency_spectrum(predictions_plot, targets_plot, best_subject_id, freq_plot_path)
        
        # Visualize full time series
        timeseries_plot_path = os.path.join(vis_dir, f'time_series_{loss_type}_{best_subject_id}.png')
        plot_time_series(predictions_plot, targets_plot, best_subject_id, timeseries_plot_path)
        
        print(f"\nVisualizations saved for {loss_type} model:")
        print(f"  FC comparison: {fc_plot_path}")
        print(f"  Frequency spectrum: {freq_plot_path}")
        print(f"  Time series: {timeseries_plot_path}")
    
    print("\n" + "="*60)
    print("All done!")
    print("="*60)


if __name__ == '__main__':
    main()

