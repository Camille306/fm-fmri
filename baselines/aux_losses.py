"""
Auxiliary losses for baselines: PSD (power spectrum) loss.
Differentiable in PyTorch for use during training.
"""

import torch
import torch.nn.functional as F


def psd_power_torch(x: torch.Tensor, dim_time: int = 1) -> torch.Tensor:
    """
    Compute power spectral density via FFT. Differentiable.
    x: (B, T, V) -> returns (B, F, V) where F = T//2 + 1.
    """
    X = torch.fft.rfft(x, dim=dim_time)
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
    MSE between log-PSD of pred and target in the 0.01–0.05 Hz BOLD band.
    pred/target: (B, T, V).  Uses bandpass mask so only the core slow-
    fluctuation band contributes to the loss.
    """
    T = pred.shape[dim_time]
    p_pred = psd_power_torch(pred, dim_time=dim_time)
    p_tgt = psd_power_torch(target, dim_time=dim_time)
    freqs = torch.fft.rfftfreq(T, d=1.0 / fs, device=pred.device, dtype=pred.dtype)
    mask = ((freqs >= low_hz) & (freqs <= high_hz)).float()
    mask = mask.unsqueeze(0).unsqueeze(-1)  # (1, F, 1) for broadcasting
    log_pred = torch.log(p_pred + eps)
    log_tgt = torch.log(p_tgt + eps)
    diff = (log_pred - log_tgt) ** 2
    masked_diff = diff * mask
    n_bins = mask.sum().clamp(min=1.0)
    return masked_diff.sum() / (n_bins * pred.shape[0] * pred.shape[2])
