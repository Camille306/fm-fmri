"""
TimeVAE Baseline Model for Rest-to-Task fMRI Prediction

This script implements a Variational Autoencoder (VAE) baseline model that predicts 
task fMRI data from resting-state fMRI data. The model uses:
- Encoder: Maps input rest sequence to latent distribution (mean, log_var)
- Decoder: Reconstructs task sequence from latent samples
- Loss: KL divergence + reconstruction (MSE) loss

Training improvements to reduce generator loss:
- Teacher forcing: Uses ground truth targets during training for better learning
- Gradient clipping: Stabilizes training by clipping gradients
- Beta warm-up (KL annealing): Gradually increases KL weight to prevent posterior collapse
- Better initialization: Improved weight initialization for decoder layers
- Adaptive teacher forcing: Gradually decreases teacher forcing ratio during training

Input:  rest window  (B, L, V)
Output: task sequence (B, T, V)
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional
from scipy import signal
from scipy.stats import pearsonr

# Hardcoded import: dataset from project root (baselines/ -> parent.parent)
import importlib.util
import sys
_baselines_dir = Path(__file__).resolve().parent
_REPO_ROOT = _baselines_dir.parent
sys.path.insert(0, str(_baselines_dir))
sys.path.insert(0, str(_REPO_ROOT))
spec = importlib.util.spec_from_file_location("dataset", str(_baselines_dir.parent / "dataset.py"))
dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_module)
HCPRestingFCDataset = dataset_module.HCPRestingFCDataset
from eval_viz import save_closest_subject_visualizations
from aux_losses import frequency_loss_torch
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
from fc_metrics import compute_fc_topk_precision_recall_auc, topk_metric_keys, TOP_K_PERCENTILES


# ============================================================================
# FMRIWindowDataset - Embedded directly to avoid import issues
# ============================================================================

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
                    window_count = 0
                    max_rest_idx = rest_time_points - self.lookback_length
                    max_task_idx = task_time_points - self.prediction_length
                    max_windows = min(max_rest_idx, max_task_idx)
                    
                    for rest_start_idx in range(0, max_windows + 1, self.stride):
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
                            'task_start_idx': None
                        })
                        
                        window_count += 1
                        if self.max_samples_per_subject and window_count >= self.max_samples_per_subject:
                            break
                
            except Exception as e:
                print(f"Warning: Failed to process subject {subject_id}: {e}")
                continue
    
    def _compute_normalization_stats(self, sample_size: int = 1000, batch_size: int = 100):
        """Compute mean and std for each variable using a sample of windows."""
        if len(self.window_metadata) == 0:
            return
        
        actual_sample_size = min(sample_size, len(self.window_metadata))
        sample_indices = np.random.choice(len(self.window_metadata), actual_sample_size, replace=False)
        
        print(f"  Computing stats from {actual_sample_size} sample windows (batch size: {batch_size})...")
        
        rest_sum = None
        rest_sum_sq = None
        rest_count = 0
        
        task_sum = None
        task_sum_sq = None
        task_count = 0
        
        num_variables = None
        
        for batch_start in range(0, len(sample_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(sample_indices))
            batch_indices = sample_indices[batch_start:batch_end]
            
            batch_rest_inputs = []
            batch_task_targets = []
            
            for idx in batch_indices:
                meta = self.window_metadata[idx]
                try:
                    rest_timeseries = self.dataset.load_subject(meta['subject_id'])
                    if len(rest_timeseries.shape) == 1:
                        rest_timeseries = rest_timeseries.reshape(-1, 1)
                    
                    if num_variables is None:
                        num_variables = rest_timeseries.shape[1]
                    
                    rest_start_idx = meta['rest_start_idx']
                    rest_end_idx = rest_start_idx + self.lookback_length
                    rest_input_seq = rest_timeseries[rest_start_idx:rest_end_idx].astype(np.float32)
                    batch_rest_inputs.append(rest_input_seq)
                    
                    if self.use_task_target:
                        task_timeseries = self.dataset.load_task_subject(meta['subject_id'])
                        if len(task_timeseries.shape) == 1:
                            task_timeseries = task_timeseries.reshape(-1, 1)
                        
                        task_start_idx = meta['task_start_idx']
                        task_end_idx = task_start_idx + self.prediction_length
                        task_target_seq = task_timeseries[task_start_idx:task_end_idx].astype(np.float32)
                        batch_task_targets.append(task_target_seq)
                    else:
                        target_idx = rest_end_idx + self.prediction_length - 1
                        if target_idx < len(rest_timeseries):
                            task_target_seq = rest_timeseries[rest_end_idx:target_idx + 1].astype(np.float32)
                            batch_task_targets.append(task_target_seq)
                        
                except Exception as e:
                    continue
            
            if len(batch_rest_inputs) > 0:
                batch_rest = np.stack(batch_rest_inputs)
                batch_rest_flat = batch_rest.reshape(-1, num_variables)
                
                if rest_sum is None:
                    rest_sum = np.sum(batch_rest_flat, axis=0)
                    rest_sum_sq = np.sum(batch_rest_flat ** 2, axis=0)
                    rest_count = len(batch_rest_flat)
                else:
                    rest_sum += np.sum(batch_rest_flat, axis=0)
                    rest_sum_sq += np.sum(batch_rest_flat ** 2, axis=0)
                    rest_count += len(batch_rest_flat)
                
                del batch_rest, batch_rest_flat, batch_rest_inputs
            
            if len(batch_task_targets) > 0:
                batch_task = np.stack(batch_task_targets)
                batch_task_flat = batch_task.reshape(-1, num_variables)
                
                if task_sum is None:
                    task_sum = np.sum(batch_task_flat, axis=0)
                    task_sum_sq = np.sum(batch_task_flat ** 2, axis=0)
                    task_count = len(batch_task_flat)
                else:
                    task_sum += np.sum(batch_task_flat, axis=0)
                    task_sum_sq += np.sum(batch_task_flat ** 2, axis=0)
                    task_count += len(batch_task_flat)
                
                del batch_task, batch_task_flat, batch_task_targets
            
            if (batch_end % 500 == 0) or (batch_end == len(sample_indices)):
                print(f"    Processed {batch_end}/{len(sample_indices)} windows...")
        
        if rest_count == 0:
            print("Warning: Could not compute normalization stats")
            return
        
        self.rest_means = rest_sum / rest_count
        rest_variance = (rest_sum_sq / rest_count) - (self.rest_means ** 2)
        self.rest_stds = np.sqrt(np.maximum(rest_variance, 0))
        self.rest_stds = np.where(self.rest_stds < 1e-8, 1.0, self.rest_stds)
        
        if task_count > 0:
            self.task_means = task_sum / task_count
            task_variance = (task_sum_sq / task_count) - (self.task_means ** 2)
            self.task_stds = np.sqrt(np.maximum(task_variance, 0))
            self.task_stds = np.where(self.task_stds < 1e-8, 1.0, self.task_stds)
        else:
            self.task_means = self.rest_means
            self.task_stds = self.rest_stds
        
        print(f"  Rest normalization stats: means shape {self.rest_means.shape}, stds shape {self.rest_stds.shape}")
        print(f"  Task normalization stats: means shape {self.task_means.shape}, stds shape {self.task_stds.shape}")
    
    def __len__(self):
        return len(self.window_metadata)
    
    def __getitem__(self, idx):
        """Load data on-demand for the requested window."""
        meta = self.window_metadata[idx]
        
        rest_timeseries = self.dataset.load_subject(meta['subject_id'])
        if len(rest_timeseries.shape) == 1:
            rest_timeseries = rest_timeseries.reshape(-1, 1)
        
        rest_start_idx = meta['rest_start_idx']
        rest_end_idx = rest_start_idx + self.lookback_length
        input_seq = rest_timeseries[rest_start_idx:rest_end_idx].astype(np.float32)
        
        if self.use_task_target:
            task_timeseries = self.dataset.load_task_subject(meta['subject_id'])
            if len(task_timeseries.shape) == 1:
                task_timeseries = task_timeseries.reshape(-1, 1)
            
            task_start_idx = meta['task_start_idx']
            task_end_idx = task_start_idx + self.prediction_length
            target_seq = task_timeseries[task_start_idx:task_end_idx].astype(np.float32)
        else:
            target_idx = rest_end_idx + self.prediction_length - 1
            target_seq = rest_timeseries[rest_end_idx:target_idx + 1].astype(np.float32)
        
        if self.normalize:
            if self.rest_means is not None and self.rest_stds is not None:
                input_seq = (input_seq - self.rest_means) / self.rest_stds
            if self.task_means is not None and self.task_stds is not None:
                target_seq = (target_seq - self.task_means) / self.task_stds
        
        input_seq = torch.from_numpy(input_seq)
        target_seq = torch.from_numpy(target_seq)
        
        task_start_idx = meta.get('task_start_idx', 0)
        return {
            'input': input_seq,
            'target': target_seq[-1] if self.prediction_length == 1 else target_seq,
            'subject_id': meta['subject_id'],
            'task_start_idx': int(task_start_idx),
        }


# ============================================================================
# Utility functions for evaluation - Embedded directly
# ============================================================================

def compute_functional_connectivity(data: np.ndarray) -> np.ndarray:
    """Compute functional connectivity matrix (correlation matrix)."""
    fc_matrix = np.corrcoef(data.T)
    fc_matrix = np.nan_to_num(fc_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
    return fc_matrix


def compute_frequency_difference(pred: np.ndarray, target: np.ndarray, fs: float = 0.72) -> float:
    """Compute frequency difference between predicted and target signals."""
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    if target.ndim == 1:
        target = target.reshape(1, -1)
    
    num_samples, num_variables = pred.shape
    freq_diffs = []
    
    for var_idx in range(num_variables):
        pred_signal = pred[:, var_idx]
        target_signal = target[:, var_idx]
        
        try:
            freqs_pred, psd_pred = signal.welch(pred_signal, fs=fs, nperseg=min(64, len(pred_signal)))
            freqs_target, psd_target = signal.welch(target_signal, fs=fs, nperseg=min(64, len(target_signal)))
            
            if len(freqs_pred) != len(freqs_target):
                min_len = min(len(freqs_pred), len(freqs_target))
                psd_pred = psd_pred[:min_len]
                psd_target = psd_target[:min_len]
            
            freq_diff = np.mean(np.abs(psd_pred - psd_target))
            freq_diffs.append(freq_diff)
        except:
            continue
    
    return np.mean(freq_diffs) if freq_diffs else 0.0


def compute_fc_similarity(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute functional connectivity similarity between predicted and target. Returns 0.0 on any failure/NaN."""
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    if target.ndim == 1:
        target = target.reshape(1, -1)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    if pred.size == 0 or target.size == 0:
        return 0.0
    try:
        fc_pred = compute_functional_connectivity(pred)
        fc_target = compute_functional_connectivity(target)
        fc_pred = np.nan_to_num(fc_pred, nan=0.0, posinf=1.0, neginf=-1.0)
        fc_target = np.nan_to_num(fc_target, nan=0.0, posinf=1.0, neginf=-1.0)
    except Exception:
        return 0.0
    mask = np.triu(np.ones_like(fc_pred, dtype=bool), k=1)
    fc_pred_vec = fc_pred[mask].astype(np.float64)
    fc_target_vec = fc_target[mask].astype(np.float64)
    if len(fc_pred_vec) < 2 or len(fc_target_vec) < 2:
        return 0.0
    std_p = np.std(fc_pred_vec)
    std_t = np.std(fc_target_vec)
    if std_p < 1e-10 or std_t < 1e-10:
        return 0.0
    try:
        corr, _ = pearsonr(fc_pred_vec, fc_target_vec)
        out = float(corr) if not (np.isnan(corr) or np.isinf(corr)) else 0.0
        return max(-1.0, min(1.0, out))
    except Exception:
        return 0.0


def aggregate_subject_timeline(chunks, starts, total_len):
    """
    Deduplicate by averaging overlaps. Subject-level: one (total_len, V) per subject.
    chunks: list of (T, V) arrays; starts: list of start indices; total_len: full timeline length.
    """
    if len(chunks) == 0:
        return None
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


def evaluate_subject_level_dedup(
    model,
    dataloader,
    device,
    prediction_length,
    beta=1.0,
    compute_freq_diff=True,
    compute_fc_sim=True,
    return_for_viz=False,
):
    """
    Subject-level evaluation: aggregate predicted/target windows per subject by task_start_idx,
    compute MSE/MAE/freq_diff/fc_similarity per subject on aggregated timeline, then report mean ± std across subjects.
    """
    from collections import defaultdict

    model.eval()
    subj_pred = defaultdict(list)
    subj_tgt = defaultdict(list)
    subj_starts = defaultdict(list)
    subj_total_len = defaultdict(int)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval (subject-level)", leave=False):
            input_seq = batch['input'].to(device).float()
            target = batch['target'].to(device).float()
            sids = batch['subject_id']
            starts = batch['task_start_idx']
            if not isinstance(sids, (list, tuple)):
                sids = list(sids)
            if not isinstance(starts, (list, tuple)):
                starts = list(starts)

            pred, mean, logvar = model(input_seq, prediction_length=prediction_length, target=None, teacher_forcing_ratio=0.0)
            if pred.dim() == 2:
                pred = pred.unsqueeze(1)
            if target.dim() == 2:
                target = target.unsqueeze(1)
            pred_np = pred.cpu().numpy()
            tgt_np = target.cpu().numpy()

            for b, sid in enumerate(sids):
                sid = str(sid)
                st = int(starts[b])
                subj_pred[sid].append(pred_np[b])
                subj_tgt[sid].append(tgt_np[b])
                subj_starts[sid].append(st)
                subj_total_len[sid] = max(subj_total_len[sid], st + prediction_length)

    per_subj = {}
    for sid in sorted(subj_pred.keys()):
        total_len = subj_total_len[sid]
        pred_full = aggregate_subject_timeline(subj_pred[sid], subj_starts[sid], total_len)
        tgt_full = aggregate_subject_timeline(subj_tgt[sid], subj_starts[sid], total_len)
        if pred_full is None or tgt_full is None:
            continue
        mse = float(np.mean((pred_full - tgt_full) ** 2))
        mae = float(np.mean(np.abs(pred_full - tgt_full)))
        freq_diff = float(compute_frequency_difference(pred_full, tgt_full)) if compute_freq_diff else 0.0
        fc_sim = float(compute_fc_similarity(pred_full, tgt_full)) if compute_fc_sim else 0.0
        fc_sim = 0.0 if (np.isnan(fc_sim) or np.isinf(fc_sim)) else fc_sim
        topk = compute_fc_topk_precision_recall_auc(pred_full, tgt_full)
        per_subj[sid] = {
            'mse': mse, 'mae': mae, 'freq_diff': freq_diff, 'fc_similarity': fc_sim,
            **{key: val for key, val in topk.items() if key != "k_percentiles" and isinstance(val, (int, float))},
        }

    metrics = {'num_subjects': len(per_subj)}
    keys = ['mse', 'mae', 'freq_diff', 'fc_similarity'] + list(topk_metric_keys())
    if len(per_subj) == 0:
        for k in keys:
            metrics[k] = float('nan')
            metrics[k + '_std'] = float('nan')
        return (metrics, subj_pred, subj_tgt, subj_starts, subj_total_len, per_subj) if return_for_viz else metrics
    for k in keys:
        vals = np.array([per_subj[s].get(k, float("nan")) for s in per_subj], dtype=np.float64)
        use_nanmean = (k == 'fc_similarity')
        m = np.nanmean(vals) if use_nanmean else np.mean(vals)
        s = np.nanstd(vals) if use_nanmean else np.std(vals)
        metrics[k] = float(m) if not (np.isnan(m) or np.isinf(m)) else 0.0
        metrics[k + '_std'] = float(s) if not (np.isnan(s) or np.isinf(s)) else 0.0
    if return_for_viz:
        return metrics, subj_pred, subj_tgt, subj_starts, subj_total_len, per_subj
    return metrics


def collect_real_and_generated_window_level(model, dataloader, device, prediction_length):
    """
    Run model on dataloader and collect all (real, generated) task windows as numpy arrays.
    Returns (X_real, X_gen) each of shape (N, T, V) for cFID-FC.
    """
    model.eval()
    reals, gens = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collect real/gen for cFID-FC", leave=False):
            input_seq = batch['input'].to(device).float()
            target = batch['target'].to(device).float()
            pred, _, _ = model(input_seq, prediction_length=prediction_length, target=None, teacher_forcing_ratio=0.0)
            if pred.dim() == 2:
                pred = pred.unsqueeze(1)
            if target.dim() == 2:
                target = target.unsqueeze(1)
            reals.append(target.cpu().numpy())
            gens.append(pred.cpu().numpy())
    X_real = np.concatenate(reals, axis=0)
    X_gen = np.concatenate(gens, axis=0)
    return X_real, X_gen


def compute_and_save_cfid_fc(model, test_loader, device, prediction_length, save_dir, max_fc_dim=500, seed=42):
    """Compute cFID-FC (conditional Fréchet distance on FC) and write cfid_fc.txt to save_dir."""
    try:
        from re_eval.fc_utils import cfid_fc
    except ImportError:
        print("  [Skip cFID-FC] re_eval.fc_utils not found; run from repo root with re_eval available.")
        return
    X_real, X_gen = collect_real_and_generated_window_level(
        model, test_loader, device, prediction_length
    )
    rng = np.random.default_rng(seed)
    cfid = cfid_fc(X_real, X_gen, eps=1e-6, max_fc_dim=max_fc_dim, rng=rng)
    out_path = Path(save_dir) / "cfid_fc.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"cfid_fc\t{cfid:.6f}\n")
    print(f"cFID-FC (Fréchet distance on FC): {cfid:.6f}  ->  {out_path}")


# ============================================================================
# TimeVAE Model Classes
# ============================================================================

class TimeVAEEncoder(nn.Module):
    """
    Encoder network for TimeVAE.
    Maps input sequence to latent distribution parameters.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        latent_dim: int = 64,
        dropout: float = 0.1,
    ):
        """
        Initialize encoder.
        
        Args:
            input_dim: Number of input features (ROIs)
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            latent_dim: Dimension of latent space
            dropout: Dropout rate
        """
        super(TimeVAEEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        
        # LSTM layers to process sequence
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Project to latent distribution parameters
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            mean: Mean of latent distribution, shape (batch_size, latent_dim)
            logvar: Log variance of latent distribution, shape (batch_size, latent_dim)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch_size, seq_len, hidden_dim)
        
        # Use the last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Project to latent space
        mean = self.fc_mean(last_hidden)  # (batch_size, latent_dim)
        logvar = self.fc_logvar(last_hidden)  # (batch_size, latent_dim)
        
        return mean, logvar


class TimeVAEDecoder(nn.Module):
    """
    Decoder network for TimeVAE.
    Reconstructs task sequence from latent code.
    Uses teacher forcing during training for better learning.
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 166,
        max_prediction_length: int = 256,
        dropout: float = 0.1,
    ):
        """
        Initialize decoder.
        
        Args:
            latent_dim: Dimension of latent space
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            output_dim: Number of output features (ROIs)
            max_prediction_length: Maximum supported prediction length
            dropout: Dropout rate
        """
        super(TimeVAEDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.max_prediction_length = max_prediction_length
        
        # Project latent code to initial hidden state (with better initialization)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.latent_to_cell = nn.Linear(latent_dim, hidden_dim * num_layers)
        
        # Initialize these layers with smaller weights
        nn.init.xavier_uniform_(self.latent_to_hidden.weight, gain=0.5)
        nn.init.zeros_(self.latent_to_hidden.bias)
        nn.init.xavier_uniform_(self.latent_to_cell.weight, gain=0.5)
        nn.init.zeros_(self.latent_to_cell.bias)
        
        # LSTM layers for sequence generation
        self.lstm = nn.LSTM(
            input_size=output_dim,  # Will use previous output as input
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Output projection with better initialization
        self.fc = nn.Linear(hidden_dim, output_dim)
        nn.init.xavier_uniform_(self.fc.weight, gain=1.0)
        nn.init.zeros_(self.fc.bias)
        
        # Learnable time embeddings for multi-step prediction
        self.time_embed = nn.Parameter(torch.zeros(max_prediction_length, output_dim))
        nn.init.normal_(self.time_embed, mean=0.0, std=0.01)  # Smaller initialization
        
    def forward(self, z, prediction_length: int = 1, target=None, teacher_forcing_ratio: float = 0.5):
        """
        Forward pass with optional teacher forcing.
        
        Args:
            z: Latent code, shape (batch_size, latent_dim)
            prediction_length: Number of time steps to predict
            target: Target sequence for teacher forcing, shape (batch_size, prediction_length, output_dim)
            teacher_forcing_ratio: Probability of using teacher forcing (0.0 = no teacher forcing, 1.0 = always)
            
        Returns:
            Output tensor of shape (batch_size, prediction_length, output_dim) if prediction_length > 1
            or (batch_size, output_dim) if prediction_length == 1
        """
        batch_size = z.shape[0]
        
        if prediction_length > self.max_prediction_length:
            raise ValueError(
                f"prediction_length={prediction_length} exceeds max_prediction_length={self.max_prediction_length}"
            )
        
        # Project latent to initial hidden and cell states
        h0 = self.latent_to_hidden(z)  # (batch_size, hidden_dim * num_layers)
        c0 = self.latent_to_cell(z)  # (batch_size, hidden_dim * num_layers)
        
        # Reshape to (num_layers, batch_size, hidden_dim)
        h0 = h0.view(self.num_layers, batch_size, self.hidden_dim)
        c0 = c0.view(self.num_layers, batch_size, self.hidden_dim)
        
        if prediction_length == 1:
            # Single step prediction
            # Use zero input with time embedding
            input_embed = self.time_embed[0:1].unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, 1, output_dim)
            lstm_out, _ = self.lstm(input_embed, (h0, c0))
            output = self.fc(lstm_out[:, -1, :])  # (batch_size, output_dim)
            return output
        else:
            # Multi-step prediction with optional teacher forcing
            outputs = []
            # Start with time embedding
            input_embed = self.time_embed[0:1].unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, 1, output_dim)
            hidden = (h0, c0)
            use_teacher_forcing = (target is not None and 
                                  self.training and 
                                  torch.rand(1).item() < teacher_forcing_ratio)
            
            for t in range(prediction_length):
                lstm_out, hidden = self.lstm(input_embed, hidden)
                output_t = self.fc(lstm_out[:, -1, :])  # (batch_size, output_dim)
                outputs.append(output_t)
                
                # Prepare next input
                if t < prediction_length - 1:
                    if use_teacher_forcing:
                        # Use ground truth target as next input
                        input_embed = target[:, t:t+1, :]  # (batch_size, 1, output_dim)
                    else:
                        # Use predicted output + time embedding
                        # Combine output with time embedding for better conditioning
                        next_time_embed = self.time_embed[t+1:t+2].unsqueeze(0).repeat(batch_size, 1, 1)
                        input_embed = 0.7 * output_t.unsqueeze(1) + 0.3 * next_time_embed
            
            return torch.stack(outputs, dim=1)  # (batch_size, prediction_length, output_dim)


class TimeVAE(nn.Module):
    """
    TimeVAE model for rest-to-task fMRI prediction.
    
    Architecture:
    - Encoder: Maps rest sequence to latent distribution
    - Reparameterization: Samples from latent distribution
    - Decoder: Reconstructs task sequence from latent code
    """
    
    def __init__(
        self,
        input_dim: int = 166,
        hidden_dim: int = 128,
        num_layers: int = 2,
        latent_dim: int = 64,
        output_dim: int = 166,
        max_prediction_length: int = 256,
        dropout: float = 0.1,
    ):
        """
        Initialize TimeVAE model.
        
        Args:
            input_dim: Number of input features (ROIs)
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            latent_dim: Dimension of latent space
            output_dim: Number of output features (ROIs)
            max_prediction_length: Maximum supported prediction length
            dropout: Dropout rate
        """
        super(TimeVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.max_prediction_length = max_prediction_length
        
        self.encoder = TimeVAEEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            latent_dim=latent_dim,
            dropout=dropout
        )
        
        self.decoder = TimeVAEDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            max_prediction_length=max_prediction_length,
            dropout=dropout
        )
    
    def reparameterize(self, mean, logvar):
        """
        Reparameterization trick to sample from latent distribution.
        
        Args:
            mean: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent code
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x, prediction_length: int = 1, target=None, teacher_forcing_ratio: float = 0.0):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            prediction_length: Number of time steps to predict
            target: Target sequence for teacher forcing (only used during training)
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            Output tensor and latent distribution parameters
        """
        # Encode to latent distribution
        mean, logvar = self.encoder(x)
        
        # Sample from latent distribution
        z = self.reparameterize(mean, logvar)
        
        # Decode to task sequence (with optional teacher forcing)
        output = self.decoder(z, prediction_length=prediction_length, 
                            target=target, teacher_forcing_ratio=teacher_forcing_ratio)
        
        return output, mean, logvar


def vae_loss(pred, target, mean, logvar, beta: float = 1.0):
    """
    VAE loss function: reconstruction loss + KL divergence.
    
    Args:
        pred: Predicted tensor
        target: Target tensor
        mean: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence term (beta-VAE)
        
    Returns:
        Total loss, reconstruction loss, KL loss
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(pred, target, reduction='mean')
    
    # KL divergence: -0.5 * sum(1 + logvar - mean^2 - exp(logvar))
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
    kl_loss = torch.mean(kl_loss)
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


def train_epoch(model, dataloader, optimizer, device, prediction_length=1, beta=1.0, 
                teacher_forcing_ratio=0.5, max_grad_norm=1.0, freq_loss_weight=0.0):
    """
    Train for one epoch with teacher forcing and gradient clipping.
    
    Args:
        model: TimeVAE model
        dataloader: Data loader
        optimizer: Optimizer
        device: Device
        prediction_length: Length of prediction
        beta: KL divergence weight
        teacher_forcing_ratio: Probability of using teacher forcing (0.0-1.0)
        max_grad_norm: Maximum gradient norm for clipping
    """
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        input_seq = batch['input'].to(device).float()  # (batch_size, seq_len, num_variables)
        target = batch['target'].to(device).float()  # (batch_size, num_variables) or (batch_size, pred_len, num_variables)
        
        # Handle target shape for teacher forcing
        if prediction_length > 1:
            if target.dim() == 2:
                target_expanded = target.unsqueeze(1).repeat(1, prediction_length, 1)
            else:
                target_expanded = target[:, :prediction_length, :] if target.shape[1] >= prediction_length else target
        else:
            target_expanded = None
        
        # Forward pass
        optimizer.zero_grad()
        
        # Encode to latent
        mean, logvar = model.encoder(input_seq)
        z = model.reparameterize(mean, logvar)
        
        # Decode with teacher forcing during training
        if prediction_length == 1:
            pred = model.decoder(z, prediction_length=1, target=None, teacher_forcing_ratio=0.0)
            # pred: (batch_size, num_variables), target: (batch_size, num_variables)
            loss, recon_loss, kl_loss = vae_loss(pred, target, mean, logvar, beta=beta)
            pred_psd = pred.unsqueeze(1)
            target_psd = target.unsqueeze(1)
        else:
            pred = model.decoder(z, prediction_length=prediction_length, 
                               target=target_expanded, teacher_forcing_ratio=teacher_forcing_ratio)
            # pred: (batch_size, prediction_length, num_variables)
            loss, recon_loss, kl_loss = vae_loss(pred, target_expanded, mean, logvar, beta=beta)
            pred_psd = pred
            target_psd = target_expanded
        if freq_loss_weight > 0:
            loss = loss + freq_loss_weight * frequency_loss_torch(pred_psd, target_psd)
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches if num_batches > 0 else 0.0,
        'recon': total_recon / num_batches if num_batches > 0 else 0.0,
        'kl': total_kl / num_batches if num_batches > 0 else 0.0,
    }


def evaluate(
    model,
    dataloader,
    device,
    prediction_length=1,
    beta=1.0,
    compute_freq_diff: bool = True,
    compute_fc_sim: bool = True
):
    """
    Evaluate model on test set with multiple metrics.
    
    Returns:
        Dictionary with metrics: mse, mae, freq_diff, fc_similarity, kl_loss
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    total_mse = 0.0
    total_mae = 0.0
    total_kl = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_seq = batch['input'].to(device).float()
            target = batch['target'].to(device).float()
            
            # Forward pass (no teacher forcing during evaluation)
            pred, mean, logvar = model(input_seq, prediction_length=prediction_length, 
                                     target=None, teacher_forcing_ratio=0.0)
            
            # Handle both single step and sequence predictions
            if prediction_length == 1:
                # pred: (batch_size, num_variables), target: (batch_size, num_variables)
                mse = F.mse_loss(pred, target)
                mae = torch.mean(torch.abs(pred - target))
            else:
                # pred: (batch_size, prediction_length, num_variables)
                # target: (batch_size, prediction_length, num_variables) or (batch_size, num_variables)
                if target.dim() == 2:
                    target = target.unsqueeze(1).repeat(1, prediction_length, 1)
                mse = F.mse_loss(pred, target)
                mae = torch.mean(torch.abs(pred - target))
            
            # KL loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
            kl_loss = torch.mean(kl_loss)
            
            total_mse += mse.item()
            total_mae += mae.item()
            total_kl += kl_loss.item()
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
        'kl_loss': total_kl / num_batches if num_batches > 0 else float('inf'),
    }
    
    # Compute frequency difference and FC similarity on aggregated data
    if compute_freq_diff or compute_fc_sim:
        predictions = np.concatenate(all_predictions, axis=0)  # (num_samples, num_variables)
        targets = np.concatenate(all_targets, axis=0)  # (num_samples, num_variables)
        
        if compute_freq_diff:
            freq_diff = compute_frequency_difference(predictions, targets)
            metrics['freq_diff'] = freq_diff
        
        if compute_fc_sim:
            fc_sim = compute_fc_similarity(predictions, targets)
            metrics['fc_similarity'] = fc_sim
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train TimeVAE Baseline for Rest-to-Task fMRI Prediction')
    
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
                       help='Number of time steps to predict (None = infer from task data, or default to 166 if task_root not provided)')
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
    parser.add_argument('--latent_dim', type=int, default=64,
                       help='Dimension of latent space')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Weight for KL divergence term (beta-VAE)')
    parser.add_argument('--beta_warmup_epochs', type=int, default=10,
                       help='Number of epochs to warm up beta from 0 to --beta (KL annealing)')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5,
                       help='Probability of using teacher forcing during training (0.0-1.0)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Maximum gradient norm for clipping')
    parser.add_argument('--freq_loss_weight', type=float, default=0.0,
                       help='Weight for PSD (power spectrum) auxiliary loss')
    
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
    parser.add_argument('--save_dir', type=str, default='./checkpoints_timevae',
                       help='Directory to save checkpoints')
    parser.add_argument('--eval_only', action='store_true',
                       help='Load best checkpoint, run test eval and save FC/PSD visualizations only (no training)')
    parser.add_argument('--max_samples_per_subject', type=int, default=None,
                       help='Maximum windows per subject (None = all)')
    parser.add_argument('--norm_sample_size', type=int, default=1000,
                       help='Number of samples to use for computing normalization stats')
    parser.add_argument('--norm_batch_size', type=int, default=100,
                       help='Batch size for computing normalization stats')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {args.device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Save directory: {args.save_dir}")
    
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
        if args.task_root is None:
            print("  Warning: task_root not specified, cannot infer from task data")
            print("  Using default: 166 (typical task sequence length for emotion task)")
            args.prediction_length = 166
        else:
            try:
                sample_subject = hcp_dataset.subject_ids[0]
                task_data = hcp_dataset.load_task_subject(sample_subject)
                if len(task_data.shape) == 1:
                    task_data = task_data.reshape(-1, 1)
                args.prediction_length = task_data.shape[0]  # Use full task sequence length
                print(f"  Inferred prediction_length: {args.prediction_length} (full task sequence)")
            except Exception as e:
                print(f"  Warning: Could not infer prediction length: {e}")
                print("  Using default: 166 (typical task sequence length)")
                args.prediction_length = 166
    
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
    model = TimeVAE(
        input_dim=num_variables,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        latent_dim=args.latent_dim,
        output_dim=num_variables,
        max_prediction_length=max(args.prediction_length, 256),
        dropout=args.dropout
    ).to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_path = os.path.join(args.save_dir, 'best_model.pth')
    if args.eval_only:
        if not os.path.isfile(best_path):
            raise FileNotFoundError(f"eval_only: checkpoint not found at {best_path}")
        checkpoint = torch.load(best_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_metrics, subj_pred, subj_tgt, subj_starts, subj_total_len, per_subj = evaluate_subject_level_dedup(
            model, test_loader, args.device,
            prediction_length=args.prediction_length,
            beta=args.beta,
            compute_freq_diff=True, compute_fc_sim=True,
            return_for_viz=True,
        )
        print("\n" + "=" * 60)
        print("TimeVAE TEST (eval_only, Subject-level)")
        print("=" * 60)
        print(f"MSE: {test_metrics['mse']:.6f} ± {test_metrics['mse_std']:.6f}")
        print(f"MAE: {test_metrics['mae']:.6f} ± {test_metrics['mae_std']:.6f}")
        print(f"PSD: {test_metrics['freq_diff']:.6f} ± {test_metrics['freq_diff_std']:.6f}")
        print(f"FC sim: {test_metrics['fc_similarity']:.6f} ± {test_metrics['fc_similarity_std']:.6f}")
        _print_topk(test_metrics)
        print("=" * 60)
        save_closest_subject_visualizations(
            per_subj, subj_pred, subj_tgt, subj_starts, subj_total_len,
            aggregate_subject_timeline,
            out_dir=args.save_dir,
            model_name="TimeVAE",
            fs=0.72,
        )
        results_path = os.path.join(args.save_dir, 'test_results.txt')
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write("Test Set Evaluation Results (Subject-Level) [BEST CKPT]\n")
            f.write("=" * 60 + "\n")
            f.write(f"MSE (mean ± std across subjects): {test_metrics['mse']:.6f} ± {test_metrics['mse_std']:.6f}\n")
            f.write(f"MAE (mean ± std across subjects): {test_metrics['mae']:.6f} ± {test_metrics['mae_std']:.6f}\n")
            f.write(f"PSD (absolute power spectrum difference): {test_metrics['freq_diff']:.6f} ± {test_metrics['freq_diff_std']:.6f}\n")
            f.write(f"Functional Connectivity Similarity: {test_metrics['fc_similarity']:.6f} ± {test_metrics['fc_similarity_std']:.6f}\n")
            _write_topk(f, test_metrics)
            f.write(f"Num subjects: {test_metrics['num_subjects']}\n")
            f.write("=" * 60 + "\n")
        print(f"Visualizations and test_results written to {args.save_dir}")
        print("Computing cFID-FC (Fréchet distance on FC)...")
        compute_and_save_cfid_fc(
            model, test_loader, args.device, args.prediction_length, args.save_dir,
            max_fc_dim=500, seed=42,
        )
        return

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop with beta warm-up (KL annealing)
    best_val_loss = float('inf')
    best_epoch = 0
    
    print("\nStarting training...")
    print(f"Beta warm-up: {args.beta_warmup_epochs} epochs (KL annealing)")
    print(f"Teacher forcing ratio: {args.teacher_forcing_ratio}")
    print(f"Max gradient norm: {args.max_grad_norm}")
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 50)
        
        # Beta warm-up (KL annealing): gradually increase beta from 0 to target
        if args.beta_warmup_epochs > 0 and epoch <= args.beta_warmup_epochs:
            current_beta = args.beta * (epoch / args.beta_warmup_epochs)
        else:
            current_beta = args.beta
        
        # Gradually decrease teacher forcing ratio (start high, decrease over time)
        if args.teacher_forcing_ratio > 0:
            # Linearly decrease from initial ratio to 0.1 over first half of training
            if epoch <= args.epochs // 2:
                current_tf_ratio = args.teacher_forcing_ratio * (1 - (epoch - 1) / (args.epochs // 2)) + 0.1 * ((epoch - 1) / (args.epochs // 2))
            else:
                current_tf_ratio = 0.1
        else:
            current_tf_ratio = 0.0
        
        print(f"Current beta: {current_beta:.4f}, Teacher forcing: {current_tf_ratio:.3f}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, args.device, 
            prediction_length=args.prediction_length, 
            beta=current_beta,
            teacher_forcing_ratio=current_tf_ratio,
            max_grad_norm=args.max_grad_norm,
            freq_loss_weight=args.freq_loss_weight,
        )
        
        # Validate: subject-level MSE (aggregate windows per subject, then mean across subjects)
        val_metrics = evaluate_subject_level_dedup(
            model, val_loader, args.device,
            prediction_length=args.prediction_length,
            beta=current_beta,
            compute_freq_diff=True, compute_fc_sim=True)
        val_loss = val_metrics['mse']
        
        # Update learning rate
        scheduler.step()
        
        print(f"Train Loss: {train_metrics['loss']:.6f} (Recon: {train_metrics['recon']:.6f}, KL: {train_metrics['kl']:.6f})")
        print(f"Val (subject-level) MSE: {val_loss:.6f} ± {val_metrics['mse_std']:.6f}, MAE: {val_metrics['mae']:.6f} ± {val_metrics['mae_std']:.6f}, FC: {val_metrics['fc_similarity']:.4f} ± {val_metrics['fc_similarity_std']:.4f}, N={val_metrics['num_subjects']}")
        print(f"LR: {scheduler.get_last_lr()[0]:.6e}")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            checkpoint_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_metrics['loss'],
                'num_variables': num_variables,
                'args': vars(args)
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
    
    # Load best model for test evaluation
    print("\nLoading best model for test evaluation...")
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set: subject-level metrics (aggregate windows per subject, then mean ± std across subjects)
    print("\nEvaluating on test set (subject-level)...")
    test_metrics, subj_pred, subj_tgt, subj_starts, subj_total_len, per_subj = evaluate_subject_level_dedup(
        model, test_loader, args.device,
        prediction_length=args.prediction_length,
        beta=args.beta,
        compute_freq_diff=True, compute_fc_sim=True,
        return_for_viz=True,
    )
    
    # Print results (all metrics are subject-level: mean ± std across subjects)
    print("\n" + "=" * 60)
    print("Test Set Evaluation Results (Subject-Level)")
    print("=" * 60)
    print(f"MSE (mean ± std across subjects): {test_metrics['mse']:.6f} ± {test_metrics['mse_std']:.6f}")
    print(f"MAE (mean ± std across subjects): {test_metrics['mae']:.6f} ± {test_metrics['mae_std']:.6f}")
    print(f"PSD (absolute power spectrum difference): {test_metrics['freq_diff']:.6f} ± {test_metrics['freq_diff_std']:.6f}")
    print(f"Functional Connectivity Similarity: {test_metrics['fc_similarity']:.6f} ± {test_metrics['fc_similarity_std']:.6f}")
    print(f"Num subjects: {test_metrics['num_subjects']}")
    print("=" * 60)

    save_closest_subject_visualizations(
        per_subj, subj_pred, subj_tgt, subj_starts, subj_total_len,
        aggregate_subject_timeline,
        out_dir=args.save_dir,
        model_name="TimeVAE",
        fs=0.72,
    )
    
    # Save test results
    results_path = os.path.join(args.save_dir, 'test_results.txt')
    with open(results_path, 'w') as f:
        f.write("Test Set Evaluation Results (Subject-Level)\n")
        f.write("=" * 60 + "\n")
        f.write(f"MSE (mean ± std across subjects): {test_metrics['mse']:.6f} ± {test_metrics['mse_std']:.6f}\n")
        f.write(f"MAE (mean ± std across subjects): {test_metrics['mae']:.6f} ± {test_metrics['mae_std']:.6f}\n")
        f.write(f"PSD (absolute power spectrum difference): {test_metrics['freq_diff']:.6f} ± {test_metrics['freq_diff_std']:.6f}\n")
        f.write(f"Functional Connectivity Similarity: {test_metrics['fc_similarity']:.6f} ± {test_metrics['fc_similarity_std']:.6f}\n")
        f.write(f"Num subjects: {test_metrics['num_subjects']}\n")
        f.write("=" * 60 + "\n")
    
    print(f"\nTest results saved to {results_path}")
    print("Computing cFID-FC (Fréchet distance on FC)...")
    compute_and_save_cfid_fc(
        model, test_loader, args.device, args.prediction_length, args.save_dir,
        max_fc_dim=500, seed=42,
    )


if __name__ == '__main__':
    main()
