"""
LSTM Baseline Model for Rest-to-Task fMRI Prediction (Emotion Task)

This script implements a simple LSTM baseline model that predicts task fMRI data
from resting-state fMRI data. It uses MSE loss for training and evaluates on
the test set using:
1. MSE (Mean Squared Error)
2. MAE (Mean Absolute Error)
3. Frequency Difference
4. Functional Connectivity Similarity
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy import signal
from scipy.stats import pearsonr
from tqdm import tqdm
import sys
from pathlib import Path

# Import dataset classes
from dataset import HCPRestingFCDataset

# Import FMRIWindowDataset from train.py
sys.path.append(str(Path(__file__).parent))
from train import FMRIWindowDataset


class LSTMBaseline(nn.Module):
    """
    Simple LSTM baseline model for multivariate time series prediction.
    
    Architecture:
    - LSTM layers to process input sequence
    - Fully connected layer to map to output dimension
    """
    
    def __init__(
        self,
        # With AAL3 parcellation this is typically 166, but we infer from data at runtime.
        input_dim: int = 166,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 166,
        dropout: float = 0.1,
        max_prediction_length: int = 256,
    ):
        """
        Initialize LSTM model.
        
        Args:
            input_dim: Number of input features (ROIs / brain regions), depends on parcellation
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            output_dim: Number of output features (ROIs / brain regions), should match task parcellation
            dropout: Dropout rate
            max_prediction_length: Maximum supported prediction length for multi-step outputs.
        """
        super(LSTMBaseline, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.max_prediction_length = max_prediction_length
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Output projection layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Learnable time embeddings so multi-step outputs are NOT constant across time.
        # Shape: (max_prediction_length, hidden_dim)
        self.time_embed = nn.Parameter(torch.zeros(max_prediction_length, hidden_dim))
        nn.init.normal_(self.time_embed, mean=0.0, std=0.02)
        
    def forward(self, x, prediction_length: int = 1):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            prediction_length: Number of time steps to predict (default: 1)
            
        Returns:
            Output tensor of shape (batch_size, prediction_length, output_dim) if prediction_length > 1
            or (batch_size, output_dim) if prediction_length == 1
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch_size, seq_len, hidden_dim)
        
        if prediction_length == 1:
            # Use the last hidden state for single step prediction
            last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
            output = self.fc(last_hidden)  # (batch_size, output_dim)
            return output
        else:
            # Multi-step prediction baseline:
            # Use last hidden state + learnable per-time-step embeddings to allow time-varying outputs.
            last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
            if prediction_length > self.max_prediction_length:
                raise ValueError(
                    f"prediction_length={prediction_length} exceeds max_prediction_length={self.max_prediction_length}. "
                    "Increase max_prediction_length when constructing the model."
                )

            # (1, pred_len, hidden) + (batch, 1, hidden) -> (batch, pred_len, hidden)
            hidden_seq = last_hidden.unsqueeze(1) + self.time_embed[:prediction_length].unsqueeze(0)
            output = self.fc(hidden_seq)  # (batch_size, prediction_length, output_dim)
            return output


class FrequencyLoss(nn.Module):
    """
    Frequency domain loss function.
    
    Computes loss in the frequency domain by comparing power spectral densities
    of predicted and target signals using Welch's method.
    """
    
    def __init__(self, fs: float = 0.72, reduction: str = 'mean'):
        """
        Initialize frequency loss.
        
        Args:
            fs: Sampling frequency in Hz (default 0.72 Hz for fMRI TR=1.39s)
            reduction: 'mean' or 'sum' for loss reduction
        """
        super(FrequencyLoss, self).__init__()
        self.fs = fs
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute frequency loss.
        
        Args:
            pred: Predicted tensor, shape (batch_size, num_variables) or (batch_size, seq_len, num_variables)
            target: Target tensor, shape (batch_size, num_variables) or (batch_size, seq_len, num_variables)
            
        Returns:
            Scalar loss value
        """
        # Convert to numpy for frequency analysis
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        # Handle sequence predictions by flattening
        if pred_np.ndim == 3:
            batch_size, seq_len, num_vars = pred_np.shape
            pred_np = pred_np.reshape(-1, num_vars)  # (batch * seq_len, num_vars)
            target_np = target_np.reshape(-1, num_vars)  # (batch * seq_len, num_vars)
        elif pred_np.ndim == 2:
            batch_size, num_vars = pred_np.shape
        else:
            raise ValueError(f"Unexpected pred shape: {pred_np.shape}")
        
        losses = []
        
        # Compute frequency loss for each sample and variable
        for sample_idx in range(pred_np.shape[0]):
            sample_losses = []
            for var_idx in range(num_vars):
                pred_signal = pred_np[sample_idx, var_idx]
                target_signal = target_np[sample_idx, var_idx]
                
                # Skip if signal is too short
                if len(pred_signal) < 4:
                    continue
                
                # Compute power spectral density using Welch's method
                try:
                    freqs_pred, psd_pred = signal.welch(
                        pred_signal, fs=self.fs, nperseg=min(64, len(pred_signal))
                    )
                    freqs_target, psd_target = signal.welch(
                        target_signal, fs=self.fs, nperseg=min(64, len(target_signal))
                    )
                    
                    # Interpolate to common frequency grid if needed
                    if len(freqs_pred) != len(freqs_target):
                        min_len = min(len(freqs_pred), len(freqs_target))
                        psd_pred = psd_pred[:min_len]
                        psd_target = psd_target[:min_len]
                    
                    # Compute mean squared difference in power
                    freq_loss = np.mean((psd_pred - psd_target) ** 2)
                    sample_losses.append(freq_loss)
                except:
                    # If frequency computation fails, skip this variable
                    continue
            
            if len(sample_losses) > 0:
                losses.append(np.mean(sample_losses))
        
        if len(losses) == 0:
            # Fallback to MSE if frequency computation fails
            return nn.functional.mse_loss(pred, target)
        
        loss_value = np.mean(losses) if self.reduction == 'mean' else np.sum(losses)
        return torch.tensor(loss_value, dtype=pred.dtype, device=pred.device, requires_grad=True)


class FrequencyLoss(nn.Module):
    """
    Frequency domain loss function for training.
    
    Computes loss in the frequency domain by comparing power spectral densities
    of predicted and target signals using FFT.
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize frequency loss.
        
        Args:
            reduction: 'mean' or 'sum' for loss reduction
        """
        super(FrequencyLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute frequency loss.
        
        Args:
            pred: Predicted tensor, shape (batch_size, num_variables) or (batch_size, seq_len, num_variables)
            target: Target tensor, shape (batch_size, num_variables) or (batch_size, seq_len, num_variables)
            
        Returns:
            Scalar loss value
        """
        # Handle sequence predictions
        if pred.dim() == 3:
            # (batch_size, seq_len, num_variables) -> (batch_size, num_variables, seq_len)
            # Transpose to have time as last dimension for FFT
            pred = pred.transpose(1, 2)  # (batch_size, num_variables, seq_len)
            target = target.transpose(1, 2)  # (batch_size, num_variables, seq_len)
            batch_size, num_vars, seq_len = pred.shape
        elif pred.dim() == 2:
            # Single time step: (batch_size, num_variables)
            # For single time step, frequency loss doesn't make sense, fall back to MSE
            return nn.functional.mse_loss(pred, target)
        else:
            raise ValueError(f"Unexpected pred shape: {pred.shape}")
        
        # Compute FFT for each variable across time dimension
        losses = []
        
        for var_idx in range(num_vars):
            pred_signal = pred[:, var_idx, :]  # (batch_size, seq_len)
            target_signal = target[:, var_idx, :]  # (batch_size, seq_len)
            
            # Compute FFT along time dimension (dim=1)
            pred_fft = torch.fft.rfft(pred_signal, dim=1)  # (batch_size, freq_bins)
            target_fft = torch.fft.rfft(target_signal, dim=1)  # (batch_size, freq_bins)
            
            # Compute power spectral density (magnitude squared)
            pred_psd = torch.abs(pred_fft) ** 2
            target_psd = torch.abs(target_fft) ** 2
            
            # Compute MSE in frequency domain (mean over batch and frequency bins)
            freq_loss = torch.mean((pred_psd - target_psd) ** 2)
            losses.append(freq_loss)
        
        # Aggregate across variables
        total_loss = torch.stack(losses)
        
        if self.reduction == 'mean':
            return torch.mean(total_loss)
        elif self.reduction == 'sum':
            return torch.sum(total_loss)
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


def compute_frequency_difference(pred: np.ndarray, target: np.ndarray, fs: float = 0.72) -> float:
    """
    Compute frequency difference between predicted and target signals.
    
    Uses Welch's method to estimate power spectral density and computes
    the mean absolute difference in power across frequency bins.
    
    Args:
        pred: Predicted signal, shape (num_samples, num_variables) or (num_variables,)
        target: Target signal, shape (num_samples, num_variables) or (num_variables,)
        fs: Sampling frequency in Hz (default 0.72 Hz for fMRI TR=1.39s)
        
    Returns:
        Mean frequency difference across all variables
    """
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    if target.ndim == 1:
        target = target.reshape(1, -1)
    
    num_samples, num_variables = pred.shape
    freq_diffs = []
    
    for var_idx in range(num_variables):
        pred_signal = pred[:, var_idx]
        target_signal = target[:, var_idx]
        
        # Compute power spectral density using Welch's method
        freqs_pred, psd_pred = signal.welch(pred_signal, fs=fs, nperseg=min(64, len(pred_signal)))
        freqs_target, psd_target = signal.welch(target_signal, fs=fs, nperseg=min(64, len(target_signal)))
        
        # Interpolate to common frequency grid if needed
        if len(freqs_pred) != len(freqs_target):
            # Use the shorter frequency array
            min_len = min(len(freqs_pred), len(freqs_target))
            psd_pred = psd_pred[:min_len]
            psd_target = psd_target[:min_len]
        
        # Compute mean absolute difference in power
        freq_diff = np.mean(np.abs(psd_pred - psd_target))
        freq_diffs.append(freq_diff)
    
    return np.mean(freq_diffs)


def compute_functional_connectivity(data: np.ndarray) -> np.ndarray:
    """
    Compute functional connectivity matrix (correlation matrix).
    
    Args:
        data: Time series data, shape (num_timepoints, num_variables)
        
    Returns:
        Functional connectivity matrix, shape (num_variables, num_variables)
    """
    # Compute Pearson correlation matrix
    fc_matrix = np.corrcoef(data.T)  # (num_variables, num_variables)
    
    # Handle NaN values (can occur if a variable has zero variance)
    fc_matrix = np.nan_to_num(fc_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return fc_matrix


def compute_fc_similarity(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute functional connectivity similarity between predicted and target.
    
    Computes correlation matrices for both and measures their similarity using
    correlation of the upper triangular elements.
    
    Args:
        pred: Predicted signal, shape (num_samples, num_variables) or (num_variables,)
        target: Target signal, shape (num_samples, num_variables) or (num_variables,)
        
    Returns:
        Correlation coefficient between FC matrices (similarity score)
    """
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    if target.ndim == 1:
        target = target.reshape(1, -1)
    
    # Compute functional connectivity matrices
    fc_pred = compute_functional_connectivity(pred)
    fc_target = compute_functional_connectivity(target)
    
    # Extract upper triangular elements (excluding diagonal)
    mask = np.triu(np.ones_like(fc_pred, dtype=bool), k=1)
    fc_pred_vec = fc_pred[mask]
    fc_target_vec = fc_target[mask]
    
    # Compute correlation between FC matrices
    if len(fc_pred_vec) > 1 and np.std(fc_pred_vec) > 1e-10 and np.std(fc_target_vec) > 1e-10:
        corr, _ = pearsonr(fc_pred_vec, fc_target_vec)
        return corr if not np.isnan(corr) else 0.0
    else:
        return 0.0


def train_epoch(model, dataloader, optimizer, criterion, device, prediction_length=1):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        input_seq = batch['input'].to(device).float()  # (batch_size, seq_len, num_variables)
        target = batch['target'].to(device).float()  # (batch_size, num_variables) or (batch_size, pred_len, num_variables)
        
        # Forward pass
        optimizer.zero_grad()
        pred = model(input_seq, prediction_length=prediction_length)
        
        # Handle both single step and sequence predictions
        if prediction_length == 1:
            # pred: (batch_size, num_variables), target: (batch_size, num_variables)
            loss = criterion(pred, target)
        else:
            # pred: (batch_size, prediction_length, num_variables)
            # target: (batch_size, prediction_length, num_variables) or (batch_size, num_variables)
            if target.dim() == 2:
                # If target is 2D, expand it to match prediction
                target = target.unsqueeze(1).repeat(1, prediction_length, 1)
            loss = criterion(pred, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate(
    model,
    dataloader,
    criterion,
    device,
    prediction_length=1,
    compute_freq_diff: bool = True,
    compute_fc_sim: bool = True
):
    """
    Evaluate model on test set with multiple metrics.
    
    Returns:
        Dictionary with metrics: mse, mae, freq_diff, fc_similarity
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    total_mse = 0.0
    total_mae = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_seq = batch['input'].to(device).float()
            target = batch['target'].to(device).float()
            
            # Forward pass
            pred = model(input_seq, prediction_length=prediction_length)
            
            # Handle both single step and sequence predictions
            if prediction_length == 1:
                # pred: (batch_size, num_variables), target: (batch_size, num_variables)
                mse = criterion(pred, target)
                mae = torch.mean(torch.abs(pred - target))
            else:
                # pred: (batch_size, prediction_length, num_variables)
                # target: (batch_size, prediction_length, num_variables) or (batch_size, num_variables)
                if target.dim() == 2:
                    target = target.unsqueeze(1).repeat(1, prediction_length, 1)
                mse = criterion(pred, target)
                mae = torch.mean(torch.abs(pred - target))
            
            total_mse += mse.item()
            total_mae += mae.item()
            num_batches += 1
            
            # Store predictions and targets for frequency/FC analysis
            # Flatten sequence predictions for analysis
            if pred.dim() == 3:
                pred_flat = pred.reshape(-1, pred.shape[-1])  # (batch * seq_len, num_variables)
            else:
                pred_flat = pred
            if target.dim() == 3:
                target_flat = target.reshape(-1, target.shape[-1])
            else:
                target_flat = target
            
            all_predictions.append(pred_flat.cpu().numpy())
            all_targets.append(target_flat.cpu().numpy())
    
    # Aggregate metrics
    metrics = {
        'mse': total_mse / num_batches if num_batches > 0 else float('inf'),
        'mae': total_mae / num_batches if num_batches > 0 else float('inf'),
    }
    
    # Compute frequency difference and FC similarity on aggregated data
    if compute_freq_diff or compute_fc_sim:
        predictions = np.concatenate(all_predictions, axis=0)  # (num_samples, num_variables)
        targets = np.concatenate(all_targets, axis=0)  # (num_samples, num_variables)
        
        if compute_freq_diff:
            # For frequency analysis, we need time series data
            # Since we're predicting single time points, we'll compute across samples
            # as a proxy (treating samples as time points)
            freq_diff = compute_frequency_difference(predictions, targets)
            metrics['freq_diff'] = freq_diff
        
        if compute_fc_sim:
            # Compute FC similarity on the full aggregated data
            fc_sim = compute_fc_similarity(predictions, targets)
            metrics['fc_similarity'] = fc_sim
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train LSTM Baseline for Rest-to-Task fMRI Prediction')
    
    # Data arguments
    parser.add_argument('--data_root', type=str,
                       default='./data/hcp-resting-fc',
                       help='Root directory containing subject folders')
    parser.add_argument('--task_root', type=str, default=None,
                       help='Root directory for task data')
    parser.add_argument('--task_name', type=str, default='emotion',
                       help='Name of the task (e.g., "emotion")')
    parser.add_argument('--lookback_length', type=int, default=512,
                       help='Number of time steps to use as input context')
    parser.add_argument('--prediction_length', type=int, default=None,
                       help='Number of time steps to predict (None = predict full task sequence)')
    parser.add_argument('--stride', type=int, default=100,
                       help='Stride for sliding window (larger = fewer samples, faster)')
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
    parser.add_argument('--save_dir', type=str, default='./checkpoints_lstm',
                       help='Directory to save checkpoints')
    parser.add_argument('--max_samples_per_subject', type=int, default=None,
                       help='Maximum windows per subject (None = all)')
    parser.add_argument('--norm_sample_size', type=int, default=1000,
                       help='Number of samples to use for computing normalization stats')
    parser.add_argument('--norm_batch_size', type=int, default=100,
                       help='Batch size for computing normalization stats')
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'frequency'],
                       help='Loss function type: mse (Mean Squared Error) or frequency (Frequency domain loss)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {args.device}")
    
    # Create save directory based on loss type
    if args.loss_type == 'frequency':
        save_dir = args.save_dir.replace('checkpoints_lstm', 'checkpoints_lstm_freq') if 'checkpoints_lstm' in args.save_dir else os.path.join(args.save_dir, 'freq_loss')
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save directory: {save_dir}")
    
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
    # If None, try to infer from task data (use a sample subject)
    if args.prediction_length is None:
        print("Prediction length not specified, inferring from task data...")
        try:
            sample_subject = hcp_dataset.subject_ids[0]
            task_data = hcp_dataset.load_task_subject(sample_subject)
            if len(task_data.shape) == 1:
                task_data = task_data.reshape(-1, 1)
            args.prediction_length = task_data.shape[0]  # Use full task sequence length
            print(f"  Inferred prediction_length: {args.prediction_length} (full task sequence)")
        except Exception as e:
            print(f"  Warning: Could not infer prediction length: {e}")
            print("  Using default: 1")
            args.prediction_length = 1
    
    print(f"Using stride: {args.stride} (creates fewer samples, faster processing)")
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
    
    # Get number of variables from first batch
    sample_batch = next(iter(train_loader))
    num_variables = sample_batch['input'].shape[2]
    print(f"Number of variables (brain regions): {num_variables}")
    
    # Create model
    model = LSTMBaseline(
        input_dim=num_variables,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_dim=num_variables,
        dropout=args.dropout
    ).to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    if args.loss_type == 'frequency':
        print("Using Frequency Loss")
        criterion = FrequencyLoss(fs=0.72, reduction='mean')
    else:
        print("Using MSE Loss")
        criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.device, 
                                prediction_length=args.prediction_length)
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, args.device,
                              prediction_length=args.prediction_length,
                              compute_freq_diff=False, compute_fc_sim=False)
        val_loss = val_metrics['mse']
        
        # Update learning rate
        scheduler.step()
        
        print(f"Train Loss (MSE): {train_loss:.6f}")
        print(f"Val Loss (MSE): {val_loss:.6f}, Val MAE: {val_metrics['mae']:.6f}")
        print(f"LR: {scheduler.get_last_lr()[0]:.6e}")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            checkpoint_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'num_variables': num_variables,
                'args': vars(args)
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
    
    # Load best model for test evaluation
    print("\nLoading best model for test evaluation...")
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set with all metrics
    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, criterion, args.device,
                           prediction_length=args.prediction_length,
                           compute_freq_diff=True, compute_fc_sim=True)
    
    # Print results
    print("\n" + "=" * 60)
    print("Test Set Evaluation Results")
    print("=" * 60)
    print(f"MSE (Mean Squared Error): {test_metrics['mse']:.6f}")
    print(f"MAE (Mean Absolute Error): {test_metrics['mae']:.6f}")
    print(f"Frequency Difference: {test_metrics.get('freq_diff', 'N/A'):.6f}")
    print(f"Functional Connectivity Similarity: {test_metrics.get('fc_similarity', 'N/A'):.6f}")
    print("=" * 60)
    
    # Save test results
    results_path = os.path.join(save_dir, 'test_results.txt')
    with open(results_path, 'w') as f:
        f.write("Test Set Evaluation Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"MSE (Mean Squared Error): {test_metrics['mse']:.6f}\n")
        f.write(f"MAE (Mean Absolute Error): {test_metrics['mae']:.6f}\n")
        f.write(f"Frequency Difference: {test_metrics.get('freq_diff', 'N/A'):.6f}\n")
        f.write(f"Functional Connectivity Similarity: {test_metrics.get('fc_similarity', 'N/A'):.6f}\n")
        f.write("=" * 60 + "\n")
    
    print(f"\nTest results saved to {results_path}")


if __name__ == '__main__':
    main()

