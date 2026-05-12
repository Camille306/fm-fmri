"""
Training script for Timer-XL model on HCP resting-state fMRI dataset.

This script implements multivariate next token prediction training for Timer-XL,
which is suitable for forecasting multivariate time series like fMRI data.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import dataset from the same directory
from dataset import HCPRestingFCDataset

try:
    from transformers import AutoModelForCausalLM, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: transformers library not found. Install with: pip install transformers==4.40.1")
    TRANSFORMERS_AVAILABLE = False


class FMRIWindowDataset(Dataset):
    """
    PyTorch Dataset wrapper for creating sliding windows from fMRI timeseries.
    
    Memory-efficient implementation that loads data on-demand instead of storing
    all windows in memory.
    
    For rest-to-task prediction:
    - Input: resting state sequence of shape (lookback_length, num_variables)
    - Target: task state sequence of shape (prediction_length, num_variables)
    
    If task data is not available, falls back to next-token prediction on rest data.
    """
    
    def __init__(
        self,
        dataset: HCPRestingFCDataset,
        lookback_length: int = 512,
        prediction_length: int = 1,
        stride: int = 1,
        normalize: bool = True,
        split: str = 'train',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        max_samples_per_subject: int = None,
        use_task_target: bool = True,
        norm_sample_size: int = 1000,
        norm_batch_size: int = 100,
    ):
        """
        Initialize the windowed dataset.
        
        Args:
            dataset: HCPRestingFCDataset instance (should have task_root if use_task_target=True)
            lookback_length: Number of time steps to use as input context from rest data
            prediction_length: Number of time steps to predict from task data
            stride: Stride for sliding window
            normalize: Whether to normalize each variable to zero mean and unit variance
            split: 'train', 'val', or 'test'
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            max_samples_per_subject: Maximum windows per subject (None = all)
            use_task_target: If True, use task data as target; if False, use next token from rest data
            norm_sample_size: Number of windows to sample for normalization stats (default: 1000)
            norm_batch_size: Batch size for computing normalization stats (default: 100, controls memory)
        """
        self.dataset = dataset
        self.lookback_length = lookback_length
        self.prediction_length = prediction_length
        self.stride = stride
        self.normalize = normalize
        self.split = split
        self.max_samples_per_subject = max_samples_per_subject
        self.use_task_target = use_task_target and (dataset.task_root is not None)
        self.norm_sample_size = norm_sample_size
        self.norm_batch_size = norm_batch_size
        
        # Store only metadata (subject_id, rest_start_idx, task_start_idx) instead of full data
        self.window_metadata = []
        self.rest_means = None
        self.rest_stds = None
        self.task_means = None
        self.task_stds = None
        
        print(f"Creating window indices for {split} split...")
        print(f"  Mode: {'Rest-to-Task prediction' if self.use_task_target else 'Next-token prediction'}")
        self._create_window_indices(train_ratio, val_ratio)
        
        if self.normalize and len(self.window_metadata) > 0:
            print("Computing normalization statistics (this may take a while)...")
            self._compute_normalization_stats(sample_size=self.norm_sample_size, batch_size=self.norm_batch_size)
        
        print(f"Created {len(self.window_metadata)} window indices for {split} split")
    
    def _create_window_indices(self, train_ratio: float, val_ratio: float):
        """Create window metadata (subject_id, rest_start_idx, task_start_idx) without loading data."""
        all_subjects = self.dataset.subject_ids
        num_subjects = len(all_subjects)
        
        # Split subjects
        train_end = int(num_subjects * train_ratio)
        val_end = train_end + int(num_subjects * val_ratio)
        
        if self.split == 'train':
            subject_ids = all_subjects[:train_end]
        elif self.split == 'val':
            subject_ids = all_subjects[train_end:val_end]
        else:  # test
            subject_ids = all_subjects[val_end:]
        
        print(f"Processing {len(subject_ids)} subjects...")
        for subject_idx, subject_id in enumerate(subject_ids):
            if (subject_idx + 1) % 100 == 0:
                print(f"  Processed {subject_idx + 1}/{len(subject_ids)} subjects...")
            
            try:
                # Load rest data to get shape
                rest_file_path = self.dataset.get_subject_path(subject_id)
                if not rest_file_path.exists():
                    continue
                
                rest_timeseries = self.dataset.load_subject(subject_id)
                if len(rest_timeseries.shape) == 1:
                    rest_timeseries = rest_timeseries.reshape(-1, 1)
                
                rest_time_points, num_variables = rest_timeseries.shape
                
                if self.use_task_target:
                    # Load task data to get shape
                    try:
                        task_timeseries = self.dataset.load_task_subject(subject_id)
                        if len(task_timeseries.shape) == 1:
                            task_timeseries = task_timeseries.reshape(-1, 1)
                        task_time_points, task_num_variables = task_timeseries.shape
                        
                        # Ensure same number of variables
                        if task_num_variables != num_variables:
                            print(f"Warning: Subject {subject_id} has mismatched variable counts: "
                                  f"rest={num_variables}, task={task_num_variables}. Skipping.")
                            continue
                    except Exception as e:
                        print(f"Warning: Could not load task data for subject {subject_id}: {e}")
                        continue
                    
                    # Create windows: use rest data as input, task data as target
                    # We'll align them by index (assuming they're already aligned)
                    window_count = 0
                    max_rest_idx = rest_time_points - self.lookback_length
                    max_task_idx = task_time_points - self.prediction_length
                    
                    # Use the minimum to ensure we have both rest input and task target
                    max_windows = min(max_rest_idx, max_task_idx)
                    
                    for rest_start_idx in range(0, max_windows + 1, self.stride):
                        # Use same index for task (assuming temporal alignment)
                        task_start_idx = rest_start_idx
                        
                        if task_start_idx + self.prediction_length > task_time_points:
                            break
                        
                        self.window_metadata.append({
                            'subject_id': subject_id,
                            'rest_start_idx': rest_start_idx,
                            'task_start_idx': task_start_idx
                        })
                        
                        window_count += 1
                        if self.max_samples_per_subject and window_count >= self.max_samples_per_subject:
                            break
                else:
                    # Fallback: next-token prediction on rest data
                    window_count = 0
                    for start_idx in range(0, rest_time_points - self.lookback_length - self.prediction_length + 1, self.stride):
                        end_idx = start_idx + self.lookback_length
                        target_idx = end_idx + self.prediction_length - 1
                        
                        if target_idx >= rest_time_points:
                            break
                        
                        self.window_metadata.append({
                            'subject_id': subject_id,
                            'rest_start_idx': start_idx,
                            'task_start_idx': None  # Not used in this mode
                        })
                        
                        window_count += 1
                        if self.max_samples_per_subject and window_count >= self.max_samples_per_subject:
                            break
                
            except Exception as e:
                print(f"Warning: Failed to process subject {subject_id}: {e}")
                continue
    
    def _compute_normalization_stats(self, sample_size: int = 1000, batch_size: int = 100):
        """
        Compute mean and std for each variable using a sample of windows.
        Memory-efficient incremental computation to avoid OOM.
        Computes separate stats for rest (input) and task (target) data.
        
        Args:
            sample_size: Total number of windows to sample for statistics
            batch_size: Number of windows to process at once (to control memory usage)
        """
        if len(self.window_metadata) == 0:
            return
        
        # Sample a subset of windows for statistics (reduce default to avoid OOM)
        actual_sample_size = min(sample_size, len(self.window_metadata))
        sample_indices = np.random.choice(len(self.window_metadata), actual_sample_size, replace=False)
        
        print(f"  Computing stats from {actual_sample_size} sample windows (batch size: {batch_size})...")
        
        # Initialize accumulators for incremental mean/std computation
        rest_sum = None
        rest_sum_sq = None
        rest_count = 0
        
        task_sum = None
        task_sum_sq = None
        task_count = 0
        
        num_variables = None
        
        # Process in batches to control memory usage
        for batch_start in range(0, len(sample_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(sample_indices))
            batch_indices = sample_indices[batch_start:batch_end]
            
            batch_rest_inputs = []
            batch_task_targets = []
            
            for idx in batch_indices:
                meta = self.window_metadata[idx]
                try:
                    # Load rest data (input)
                    rest_timeseries = self.dataset.load_subject(meta['subject_id'])
                    if len(rest_timeseries.shape) == 1:
                        rest_timeseries = rest_timeseries.reshape(-1, 1)
                    
                    if num_variables is None:
                        num_variables = rest_timeseries.shape[1]
                    
                    rest_start_idx = meta['rest_start_idx']
                    rest_end_idx = rest_start_idx + self.lookback_length
                    rest_input_seq = rest_timeseries[rest_start_idx:rest_end_idx].astype(np.float32)
                    batch_rest_inputs.append(rest_input_seq)
                    
                    # Load task data (target) if using task target
                    if self.use_task_target:
                        task_timeseries = self.dataset.load_task_subject(meta['subject_id'])
                        if len(task_timeseries.shape) == 1:
                            task_timeseries = task_timeseries.reshape(-1, 1)
                        
                        task_start_idx = meta['task_start_idx']
                        task_end_idx = task_start_idx + self.prediction_length
                        task_target_seq = task_timeseries[task_start_idx:task_end_idx].astype(np.float32)
                        batch_task_targets.append(task_target_seq)
                    else:
                        # Use rest data as target (next token)
                        target_idx = rest_end_idx + self.prediction_length - 1
                        if target_idx < len(rest_timeseries):
                            task_target_seq = rest_timeseries[rest_end_idx:target_idx + 1].astype(np.float32)
                            batch_task_targets.append(task_target_seq)
                        
                except Exception as e:
                    continue
            
            # Process batch for rest data
            if len(batch_rest_inputs) > 0:
                batch_rest = np.stack(batch_rest_inputs)  # (batch_size, lookback_length, num_variables)
                batch_rest_flat = batch_rest.reshape(-1, num_variables)  # (batch_size * lookback_length, num_variables)
                
                if rest_sum is None:
                    rest_sum = np.sum(batch_rest_flat, axis=0)
                    rest_sum_sq = np.sum(batch_rest_flat ** 2, axis=0)
                    rest_count = len(batch_rest_flat)
                else:
                    rest_sum += np.sum(batch_rest_flat, axis=0)
                    rest_sum_sq += np.sum(batch_rest_flat ** 2, axis=0)
                    rest_count += len(batch_rest_flat)
                
                # Clear batch from memory
                del batch_rest, batch_rest_flat, batch_rest_inputs
            
            # Process batch for task data
            if len(batch_task_targets) > 0:
                batch_task = np.stack(batch_task_targets)  # (batch_size, prediction_length, num_variables)
                batch_task_flat = batch_task.reshape(-1, num_variables)  # (batch_size * prediction_length, num_variables)
                
                if task_sum is None:
                    task_sum = np.sum(batch_task_flat, axis=0)
                    task_sum_sq = np.sum(batch_task_flat ** 2, axis=0)
                    task_count = len(batch_task_flat)
                else:
                    task_sum += np.sum(batch_task_flat, axis=0)
                    task_sum_sq += np.sum(batch_task_flat ** 2, axis=0)
                    task_count += len(batch_task_flat)
                
                # Clear batch from memory
                del batch_task, batch_task_flat, batch_task_targets
            
            if (batch_end % 500 == 0) or (batch_end == len(sample_indices)):
                print(f"    Processed {batch_end}/{len(sample_indices)} windows...")
        
        if rest_count == 0:
            print("Warning: Could not compute normalization stats")
            return
        
        # Compute final statistics from accumulators
        self.rest_means = rest_sum / rest_count  # (num_variables,)
        rest_variance = (rest_sum_sq / rest_count) - (self.rest_means ** 2)
        self.rest_stds = np.sqrt(np.maximum(rest_variance, 0))  # (num_variables,)
        self.rest_stds = np.where(self.rest_stds < 1e-8, 1.0, self.rest_stds)
        
        # Compute task statistics
        if task_count > 0:
            self.task_means = task_sum / task_count  # (num_variables,)
            task_variance = (task_sum_sq / task_count) - (self.task_means ** 2)
            self.task_stds = np.sqrt(np.maximum(task_variance, 0))  # (num_variables,)
            self.task_stds = np.where(self.task_stds < 1e-8, 1.0, self.task_stds)
        else:
            # Fallback: use rest stats for task if no task data
            self.task_means = self.rest_means
            self.task_stds = self.rest_stds
        
        print(f"  Rest normalization stats: means shape {self.rest_means.shape}, stds shape {self.rest_stds.shape}")
        print(f"  Task normalization stats: means shape {self.task_means.shape}, stds shape {self.task_stds.shape}")
    
    def __len__(self):
        return len(self.window_metadata)
    
    def __getitem__(self, idx):
        """Load data on-demand for the requested window."""
        meta = self.window_metadata[idx]
        
        # Load rest data (input)
        rest_timeseries = self.dataset.load_subject(meta['subject_id'])
        if len(rest_timeseries.shape) == 1:
            rest_timeseries = rest_timeseries.reshape(-1, 1)
        
        rest_start_idx = meta['rest_start_idx']
        rest_end_idx = rest_start_idx + self.lookback_length
        input_seq = rest_timeseries[rest_start_idx:rest_end_idx].astype(np.float32)  # (lookback_length, num_variables)
        
        # Load task data (target) if using task target
        if self.use_task_target:
            task_timeseries = self.dataset.load_task_subject(meta['subject_id'])
            if len(task_timeseries.shape) == 1:
                task_timeseries = task_timeseries.reshape(-1, 1)
            
            task_start_idx = meta['task_start_idx']
            task_end_idx = task_start_idx + self.prediction_length
            target_seq = task_timeseries[task_start_idx:task_end_idx].astype(np.float32)  # (prediction_length, num_variables)
        else:
            # Fallback: use rest data as target (next token)
            target_idx = rest_end_idx + self.prediction_length - 1
            target_seq = rest_timeseries[rest_end_idx:target_idx + 1].astype(np.float32)  # (prediction_length, num_variables)
        
        # Normalize if needed
        if self.normalize:
            if self.rest_means is not None and self.rest_stds is not None:
                input_seq = (input_seq - self.rest_means) / self.rest_stds
            if self.task_means is not None and self.task_stds is not None:
                target_seq = (target_seq - self.task_means) / self.task_stds
        
        # Convert to tensors
        input_seq = torch.from_numpy(input_seq)  # (lookback_length, num_variables)
        target_seq = torch.from_numpy(target_seq)  # (prediction_length, num_variables)
        
        return {
            'input': input_seq,  # (lookback_length, num_variables) - rest data
            'target': target_seq[-1] if self.prediction_length == 1 else target_seq,  # (num_variables,) or (prediction_length, num_variables)
            'subject_id': meta['subject_id']
        }


class MultivariateProjectionHead(nn.Module):
    """Projection head to map Timer-XL univariate output to multivariate predictions."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, hidden_dim) or (batch_size, seq_len, hidden_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim) or (batch_size, seq_len, output_dim)
        """
        if x.dim() == 3:
            # Sequence input: (batch_size, seq_len, hidden_dim)
            batch_size, seq_len, hidden_dim = x.shape
            x_flat = x.reshape(-1, hidden_dim)  # (batch_size * seq_len, hidden_dim)
            out_flat = self.projection(x_flat)  # (batch_size * seq_len, output_dim)
            output_dim = out_flat.shape[-1]
            return out_flat.reshape(batch_size, seq_len, output_dim)  # (batch_size, seq_len, output_dim)
        else:
            # Single step: (batch_size, hidden_dim)
            return self.projection(x)  # (batch_size, output_dim)


def create_model_from_huggingface(model_name: str = 'thuml/timer-base-84m', device: str = 'cuda', num_variables: int = 268):
    """
    Load Timer-XL model from HuggingFace and add projection head for multivariate output.
    
    Args:
        model_name: HuggingFace model name
        device: Device to load model on
        num_variables: Number of output variables (brain regions)
    
    Returns:
        Tuple of (base_model, projection_head)
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library is required. Install with: pip install transformers==4.40.1")
    
    print(f"Loading model from HuggingFace: {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    base_model = base_model.to(device)
    
    # Create projection head - we'll determine input_dim after first forward pass
    # For now, create a placeholder that we'll replace
    projection_head = None
    
    return base_model, projection_head


def forward_with_projection(base_model, projection_head, input_flat, num_variables, device, prediction_length=1):
    """
    Forward pass through base model and projection head.
    
    Args:
        base_model: Timer-XL base model
        projection_head: Projection head module
        input_flat: Input tensor of shape (batch_size, input_seq_len)
        num_variables: Number of output variables
        device: Device to run on
        prediction_length: Number of time steps to predict (default: 1)
    
    Returns:
        predictions: (batch_size, num_variables) if prediction_length == 1
                    (batch_size, prediction_length, num_variables) if prediction_length > 1
    """
    # Get base model output
    outputs = base_model(input_flat, return_dict=True)
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs.last_hidden_state
    
    # Handle different output shapes
    if logits.dim() == 3:  # (batch_size, seq_len, hidden_dim)
        if prediction_length == 1:
            # Single step prediction: use last time step
            hidden = logits[:, -1, :]  # (batch_size, hidden_dim)
        else:
            # Multi-step prediction: use last N time steps from output
            # If we have enough time steps, use them; otherwise use autoregressive generation
            if logits.shape[1] >= prediction_length:
                # Use the last prediction_length time steps
                hidden = logits[:, -prediction_length:, :]  # (batch_size, prediction_length, hidden_dim)
            else:
                # Not enough time steps in output, use autoregressive generation
                # Start with last hidden state
                hidden_seq = []
                current_hidden = logits[:, -1, :]  # (batch_size, hidden_dim)
                
                for _ in range(prediction_length):
                    # Project current hidden state
                    if projection_head is not None:
                        pred_step = projection_head(current_hidden)  # (batch_size, num_variables)
                    else:
                        pred_step = current_hidden[:, :num_variables] if current_hidden.shape[1] >= num_variables else current_hidden
                    
                    hidden_seq.append(current_hidden.unsqueeze(1))  # (batch_size, 1, hidden_dim)
                    
                    # For autoregressive: use the prediction as input for next step
                    # Flatten and feed back (simplified approach)
                    # In practice, you might want to use a decoder or more sophisticated approach
                    # For now, we'll just repeat the hidden state (simple baseline)
                    # TODO: Implement proper autoregressive generation with decoder
                    current_hidden = current_hidden  # Keep same hidden state (simple baseline)
                
                hidden = torch.cat(hidden_seq, dim=1)  # (batch_size, prediction_length, hidden_dim)
                
    elif logits.dim() == 2:  # (batch_size, output_dim)
        if prediction_length == 1:
            hidden = logits
        else:
            # For 2D output, repeat for sequence prediction (simple baseline)
            hidden = logits.unsqueeze(1).repeat(1, prediction_length, 1)  # (batch_size, prediction_length, hidden_dim)
    else:
        raise ValueError(f"Unexpected logits shape: {logits.shape}")
    
    # Project to number of variables
    if projection_head is not None:
        pred = projection_head(hidden)  # (batch_size, num_variables) or (batch_size, prediction_length, num_variables)
    else:
        # Fallback: if no projection head, try to extract or pad
        if hidden.dim() == 3:
            # Sequence: (batch_size, prediction_length, hidden_dim)
            if hidden.shape[2] >= num_variables:
                pred = hidden[:, :, :num_variables]
            else:
                pred = torch.zeros(hidden.shape[0], hidden.shape[1], num_variables, device=device)
                pred[:, :, :hidden.shape[2]] = hidden
        else:
            # Single step: (batch_size, hidden_dim)
            if hidden.shape[1] >= num_variables:
                pred = hidden[:, :num_variables]
            else:
                pred = torch.zeros(hidden.shape[0], num_variables, device=device)
                pred[:, :hidden.shape[1]] = hidden
    
    return pred


def train_epoch(base_model, projection_head, dataloader, optimizer, criterion, device, num_variables, clip_grad_norm: float = 1.0, prediction_length: int = 1):
    """Train for one epoch."""
    base_model.train()
    if projection_head is not None:
        projection_head.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Ensure inputs/targets are float32 to match model parameters
        input_seq = batch['input'].to(device).float()  # (batch_size, lookback_length, num_variables)
        target = batch['target'].to(device).float()  # (batch_size, num_variables) or (batch_size, prediction_length, num_variables)
        
        batch_size, lookback_length, _ = input_seq.shape
        
        # Reshape to (batch_size, lookback_length * num_variables)
        # Timer-XL expects univariate input, so we flatten multivariate data
        input_flat = input_seq.reshape(batch_size, -1)
        
        # Forward pass through model and projection
        pred = forward_with_projection(base_model, projection_head, input_flat, num_variables, device, prediction_length=prediction_length)
        
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
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if clip_grad_norm > 0:
            # Clip gradients for both base model and projection head
            all_params = list(base_model.parameters())
            if projection_head is not None:
                all_params.extend(list(projection_head.parameters()))
            torch.nn.utils.clip_grad_norm_(all_params, clip_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.6f}")
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def visualize_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    inputs: np.ndarray,
    save_dir: str,
    epoch: int,
    num_samples_to_plot: int = 10,
    normalization_stats: dict = None
):
    """
    Generate visualization plots for each brain ROI.
    
    Args:
        predictions: Predicted values, shape (num_samples, num_variables)
        targets: Ground truth values, shape (num_samples, num_variables)
        inputs: Input sequences, shape (num_samples, lookback_length, num_variables)
        save_dir: Directory to save figures
        epoch: Current epoch number
        num_samples_to_plot: Number of samples to visualize per ROI
        normalization_stats: Tuple of (means, stds) for denormalization if data was normalized
    """
    num_variables = predictions.shape[1]
    num_samples = min(num_samples_to_plot, predictions.shape[0])
    
    # Denormalize if needed
    if normalization_stats is not None:
        if isinstance(normalization_stats, dict):
            # Separate normalization for rest (input) and task (target)
            rest_means, rest_stds = normalization_stats.get('rest', (None, None))
            task_means, task_stds = normalization_stats.get('task', (None, None))
            if rest_means is not None and rest_stds is not None:
                inputs = inputs * rest_stds + rest_means
            if task_means is not None and task_stds is not None:
                predictions = predictions * task_stds + task_means
                targets = targets * task_stds + task_means
        else:
            # Fallback: single normalization stats
            means, stds = normalization_stats
            predictions = predictions * stds + means
            targets = targets * stds + means
            inputs = inputs * stds + means
    
    # Create visualization directory
    vis_dir = os.path.join(save_dir, f'visualizations_epoch_{epoch}')
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"\nGenerating visualizations for {num_variables} brain ROIs...")
    print(f"Saving to: {vis_dir}")
    
    # Generate one plot per ROI
    for roi_idx in tqdm(range(num_variables), desc="Creating ROI plots"):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot multiple samples for this ROI
        for sample_idx in range(num_samples):
            lookback_length = inputs.shape[1]
            time_steps_input = np.arange(lookback_length)
            time_steps_pred = np.arange(lookback_length, lookback_length + 1)
            
            # Plot input sequence (context)
            ax.plot(
                time_steps_input,
                inputs[sample_idx, :, roi_idx],
                color='blue',
                alpha=0.3,
                linewidth=0.8,
                label='Input Context' if sample_idx == 0 else ''
            )
            
            # Plot ground truth (target)
            ax.scatter(
                time_steps_pred,
                targets[sample_idx, roi_idx],
                color='green',
                marker='o',
                s=50,
                alpha=0.7,
                zorder=5,
                label='Ground Truth' if sample_idx == 0 else ''
            )
            
            # Plot prediction
            ax.scatter(
                time_steps_pred,
                predictions[sample_idx, roi_idx],
                color='red',
                marker='x',
                s=50,
                linewidths=2,
                alpha=0.7,
                zorder=5,
                label='Prediction' if sample_idx == 0 else ''
            )
        
        # Add vertical line separating input and prediction
        ax.axvline(x=lookback_length - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Calculate metrics for this ROI
        mse = np.mean((predictions[:, roi_idx] - targets[:, roi_idx]) ** 2)
        mae = np.mean(np.abs(predictions[:, roi_idx] - targets[:, roi_idx]))
        rmse = np.sqrt(mse)
        
        # Add title and labels
        ax.set_title(
            f'Brain ROI {roi_idx + 1}/{num_variables} (Rest→Task Prediction)\n'
            f'MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}',
            fontsize=12,
            fontweight='bold'
        )
        ax.set_xlabel('Time Step (Rest Context → Task Prediction)', fontsize=11)
        ax.set_ylabel('Signal Amplitude', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(vis_dir, f'ROI_{roi_idx + 1:03d}.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"✓ Saved {num_variables} visualization plots to {vis_dir}")
    
    # Create a summary plot showing all ROIs
    create_summary_plot(predictions, targets, save_dir, epoch, normalization_stats)


def create_summary_plot(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_dir: str,
    epoch: int,
    normalization_stats: dict = None
):
    """Create a summary plot showing prediction vs target for all ROIs."""
    if normalization_stats is not None:
        if isinstance(normalization_stats, dict):
            task_means, task_stds = normalization_stats.get('task', (None, None))
            if task_means is not None and task_stds is not None:
                predictions = predictions * task_stds + task_means
                targets = targets * task_stds + task_means
        else:
            means, stds = normalization_stats
            predictions = predictions * stds + means
            targets = targets * stds + means
    
    # Calculate per-ROI metrics
    mse_per_roi = np.mean((predictions - targets) ** 2, axis=0)
    mae_per_roi = np.mean(np.abs(predictions - targets), axis=0)
    rmse_per_roi = np.sqrt(mse_per_roi)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Prediction vs Target scatter (all ROIs)
    ax1 = axes[0, 0]
    for roi_idx in range(predictions.shape[1]):
        ax1.scatter(
            targets[:, roi_idx],
            predictions[:, roi_idx],
            alpha=0.3,
            s=10
        )
    # Add diagonal line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('Ground Truth', fontsize=11)
    ax1.set_ylabel('Prediction', fontsize=11)
    ax1.set_title('Prediction vs Ground Truth (All ROIs)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: MSE per ROI
    ax2 = axes[0, 1]
    ax2.bar(range(len(mse_per_roi)), mse_per_roi, alpha=0.7, color='steelblue')
    ax2.set_xlabel('ROI Index', fontsize=11)
    ax2.set_ylabel('MSE', fontsize=11)
    ax2.set_title('Mean Squared Error per ROI', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: MAE per ROI
    ax3 = axes[1, 0]
    ax3.bar(range(len(mae_per_roi)), mae_per_roi, alpha=0.7, color='coral')
    ax3.set_xlabel('ROI Index', fontsize=11)
    ax3.set_ylabel('MAE', fontsize=11)
    ax3.set_title('Mean Absolute Error per ROI', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: RMSE per ROI
    ax4 = axes[1, 1]
    ax4.bar(range(len(rmse_per_roi)), rmse_per_roi, alpha=0.7, color='mediumseagreen')
    ax4.set_xlabel('ROI Index', fontsize=11)
    ax4.set_ylabel('RMSE', fontsize=11)
    ax4.set_title('Root Mean Squared Error per ROI', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(
        f'Summary Statistics - Epoch {epoch} (Rest→Task Prediction)\n'
        f'Overall MSE: {mse_per_roi.mean():.4f}, '
        f'Overall MAE: {mae_per_roi.mean():.4f}, '
        f'Overall RMSE: {rmse_per_roi.mean():.4f}',
        fontsize=14,
        fontweight='bold',
        y=0.995
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save summary plot
    summary_path = os.path.join(save_dir, f'visualizations_epoch_{epoch}', 'summary_statistics.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Saved summary statistics plot to {summary_path}")


def validate(base_model, projection_head, dataloader, criterion, device, num_variables, return_predictions=False, prediction_length: int = 1):
    """
    Validate the model.
    
    Args:
        base_model: Base Timer-XL model
        projection_head: Projection head for multivariate output
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to run on
        num_variables: Number of output variables
        return_predictions: If True, return predictions and targets for visualization
        prediction_length: Number of time steps to predict
    
    Returns:
        Average validation loss, and optionally (predictions, targets, inputs)
    """
    base_model.eval()
    if projection_head is not None:
        projection_head.eval()
    total_loss = 0.0
    num_batches = 0
    
    all_predictions = []
    all_targets = []
    all_inputs = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Ensure inputs/targets are float32 to match model parameters
            input_seq = batch['input'].to(device).float()
            target = batch['target'].to(device).float()
            
            batch_size, lookback_length, _ = input_seq.shape
            input_flat = input_seq.reshape(batch_size, -1)
            
            try:
                pred = forward_with_projection(base_model, projection_head, input_flat, num_variables, device, prediction_length=prediction_length)
                
                # Handle both single step and sequence predictions
                if prediction_length == 1:
                    loss = criterion(pred, target)
                else:
                    if target.dim() == 2:
                        target = target.unsqueeze(1).repeat(1, prediction_length, 1)
                    loss = criterion(pred, target)
                
                total_loss += loss.item()
                num_batches += 1
                
                if return_predictions:
                    # Flatten sequence predictions for storage
                    if pred.dim() == 3:
                        pred_flat = pred.reshape(-1, pred.shape[-1])
                    else:
                        pred_flat = pred
                    if target.dim() == 3:
                        target_flat = target.reshape(-1, target.shape[-1])
                    else:
                        target_flat = target
                    
                    all_predictions.append(pred_flat.cpu().numpy())
                    all_targets.append(target_flat.cpu().numpy())
                    all_inputs.append(input_seq.cpu().numpy())
                    
            except Exception as e:
                print(f"Warning: Validation forward pass failed: {e}")
                continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    
    if return_predictions:
        predictions = np.concatenate(all_predictions, axis=0)  # (num_samples, num_variables)
        targets = np.concatenate(all_targets, axis=0)  # (num_samples, num_variables)
        inputs = np.concatenate(all_inputs, axis=0)  # (num_samples, lookback_length, num_variables)
        return avg_loss, predictions, targets, inputs
    
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train Timer-XL on HCP resting-state fMRI data')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, 
                       default='./data/hcp-resting-fc',
                       help='Root directory containing subject folders')
    parser.add_argument('--task_root', type=str, default=None,
                       help='Root directory for task data (if None, uses next-token prediction on rest data)')
    parser.add_argument('--task_name', type=str, default='emotion',
                       help='Name of the task (e.g., "emotion")')
    parser.add_argument('--lookback_length', type=int, default=512,
                       help='Number of time steps to use as input context from rest data')
    parser.add_argument('--prediction_length', type=int, default=None,
                       help='Number of time steps to predict from task data (None = predict full task sequence)')
    parser.add_argument('--stride', type=int, default=100,
                       help='Stride for sliding window (larger = fewer samples, faster)')
    parser.add_argument('--normalize', action='store_true', default=True,
                       help='Normalize data to zero mean and unit variance')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Proportion of data for training')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Proportion of data for validation')
    parser.add_argument('--use_task_target', action='store_true', default=True,
                       help='Use task data as target (rest-to-task prediction). If False, uses next-token prediction on rest data.')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='thuml/timer-base-84m',
                       help='HuggingFace model name or path to local model')
    parser.add_argument('--from_pretrained', action='store_true', default=True,
                       help='Load pretrained model from HuggingFace')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay for optimizer')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0,
                       help='Gradient clipping norm (0 to disable)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu). Auto-detect if not specified')
    parser.add_argument('--num_workers', type=int, default=1,
                       help='Number of data loader workers')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Generate visualization plots for best model')
    parser.add_argument('--num_vis_samples', type=int, default=10,
                       help='Number of samples to visualize per ROI')
    parser.add_argument('--max_samples_per_subject', type=int, default=None,
                       help='Maximum windows per subject (None = all, helps reduce memory)')
    parser.add_argument('--norm_sample_size', type=int, default=1000,
                       help='Number of samples to use for computing normalization stats (reduced to avoid OOM)')
    parser.add_argument('--norm_batch_size', type=int, default=100,
                       help='Batch size for computing normalization stats (controls memory usage)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {args.device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load dataset
    print("Loading HCP resting-state fMRI dataset...")
    if args.task_root:
        print(f"  Task root: {args.task_root}")
        print(f"  Task name: {args.task_name}")
        print("  Mode: Rest-to-Task prediction")
    else:
        print("  Mode: Next-token prediction (no task data)")
    
    hcp_dataset = HCPRestingFCDataset(
        data_root=args.data_root,
        task_root=args.task_root,
        task_name=args.task_name
    )
    print(f"Found {len(hcp_dataset)} subjects")
    
    # Determine prediction length
    # If None, try to infer from task data (use a sample subject)
    if args.prediction_length is None and args.task_root is not None:
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
    elif args.prediction_length is None:
        # No task root, use default for next-token prediction
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
        use_task_target=args.use_task_target and (args.task_root is not None),
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
        use_task_target=args.use_task_target and (args.task_root is not None),
        norm_sample_size=args.norm_sample_size,
        norm_batch_size=args.norm_batch_size
    )
    
    # Use normalization stats from training set for validation
    if args.normalize:
        if train_dataset.rest_means is not None:
            val_dataset.rest_means = train_dataset.rest_means
            val_dataset.rest_stds = train_dataset.rest_stds
        if train_dataset.task_means is not None:
            val_dataset.task_means = train_dataset.task_means
            val_dataset.task_stds = train_dataset.task_stds
    
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
    
    # Get number of variables from first batch
    sample_batch = next(iter(train_loader))
    num_variables = sample_batch['input'].shape[2]
    print(f"Number of variables (brain regions): {num_variables}")
    
    # Create model
    if args.from_pretrained and TRANSFORMERS_AVAILABLE:
        base_model, _ = create_model_from_huggingface(args.model_name, args.device, num_variables)
    else:
        raise NotImplementedError(
            "Model creation from scratch not implemented. "
            "Please use --from_pretrained with a HuggingFace model, "
            "or implement custom model initialization."
        )
    
    # Determine model output dimension and create projection head
    print("Determining model output dimension...")
    base_model.eval()
    with torch.no_grad():
        # Ensure sample input is float32 to match model parameters
        sample_input = sample_batch['input'][:1].to(args.device).float()  # (1, lookback_length, num_variables)
        batch_size, lookback_length, _ = sample_input.shape
        sample_input_flat = sample_input.reshape(batch_size, -1)  # (1, lookback_length * num_variables)
        
        test_output = base_model(sample_input_flat, return_dict=True)
        test_logits = test_output.logits if hasattr(test_output, 'logits') else test_output.last_hidden_state
        
        if test_logits.dim() == 3:
            # Take the last time step's hidden state dimension
            model_output_dim = test_logits.shape[2]  # (batch, seq, hidden_dim)
        elif test_logits.dim() == 2:
            model_output_dim = test_logits.shape[1]  # (batch, output_dim)
        else:
            raise ValueError(f"Unexpected model output shape: {test_logits.shape}")
        
        print(f"Model output dimension: {model_output_dim}")
        print(f"Creating projection head: {model_output_dim} -> {num_variables}")
    
    # Create projection head
    projection_head = MultivariateProjectionHead(model_output_dim, num_variables).to(args.device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    
    # Combine parameters from both base model and projection head
    all_params = list(base_model.parameters()) + list(projection_head.parameters())
    optimizer = AdamW(
        all_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    normalization_stats = None
    model_output_dim = None  # Will be set during model creation
    
    # Store normalization stats if needed (for visualization)
    normalization_stats = None
    if args.normalize:
        if train_dataset.rest_means is not None and train_dataset.task_means is not None:
            normalization_stats = {
                'rest': (train_dataset.rest_means, train_dataset.rest_stds),
                'task': (train_dataset.task_means, train_dataset.task_stds)
            }
        elif train_dataset.rest_means is not None:
            normalization_stats = {
                'rest': (train_dataset.rest_means, train_dataset.rest_stds),
                'task': (train_dataset.rest_means, train_dataset.rest_stds)  # Fallback
            }
    
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_loss = train_epoch(
            base_model, projection_head, train_loader, optimizer, criterion, 
            args.device, num_variables, args.clip_grad_norm, prediction_length=args.prediction_length
        )
        
        # Validate
        # Check if we might get a new best model (for efficiency, only get predictions when needed)
        if args.visualize:
            # Always get predictions, but only visualize when we have a new best
            val_loss, predictions, targets, inputs = validate(
                base_model, projection_head, val_loader, criterion, args.device, 
                num_variables, return_predictions=True, prediction_length=args.prediction_length
            )
        else:
            val_loss = validate(
                base_model, projection_head, val_loader, criterion, args.device, 
                num_variables, return_predictions=False, prediction_length=args.prediction_length
            )
            predictions = None
            targets = None
            inputs = None
        
        # Update learning rate
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6e}")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            checkpoint_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'base_model_state_dict': base_model.state_dict(),
                'projection_head_state_dict': projection_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'model_output_dim': model_output_dim,
                'num_variables': num_variables,
                'args': vars(args)
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
            
            # Generate visualizations for best model
            if args.visualize and predictions is not None:
                print("\nGenerating visualizations for best model...")
                visualize_predictions(
                    predictions=predictions,
                    targets=targets,
                    inputs=inputs,
                    save_dir=args.save_dir,
                    epoch=epoch,
                    num_samples_to_plot=args.num_vis_samples,
                    normalization_stats=normalization_stats
                )
        
        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'base_model_state_dict': base_model.state_dict(),
                'projection_head_state_dict': projection_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'model_output_dim': model_output_dim,
                'num_variables': num_variables,
                'args': vars(args)
            }, checkpoint_path)
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
    
    # Generate final visualizations if not already done at best epoch
    if args.visualize and best_epoch > 0:
        # Check if visualizations already exist for best epoch
        vis_dir = os.path.join(args.save_dir, f'visualizations_epoch_{best_epoch}')
        if not os.path.exists(vis_dir):
            print("\nGenerating final visualizations for best model...")
            val_loss, predictions, targets, inputs = validate(
                base_model, projection_head, val_loader, criterion, args.device, 
                num_variables, return_predictions=True, prediction_length=args.prediction_length
            )
            visualize_predictions(
                predictions=predictions,
                targets=targets,
                inputs=inputs,
                save_dir=args.save_dir,
                epoch=best_epoch,
                num_samples_to_plot=args.num_vis_samples,
                normalization_stats=normalization_stats
            )


if __name__ == '__main__':
    main()

