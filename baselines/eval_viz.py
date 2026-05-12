"""
Shared evaluation visualizations for baselines.
Produces FC (functional connectome) and PSD (power spectrum) plots for the closest subject (by MSE).
"""

import os
import numpy as np
from scipy import signal
from scipy.stats import pearsonr


def compute_functional_connectivity(data: np.ndarray) -> np.ndarray:
    """Correlation matrix over time (columns = ROIs). data: (T, V) -> (V, V)."""
    fc = np.corrcoef(data.T)
    return np.nan_to_num(fc, nan=0.0, posinf=1.0, neginf=-1.0)


def compute_fc_similarity(pred: np.ndarray, target: np.ndarray) -> float:
    """Pearson correlation between upper-triangle of pred FC and target FC."""
    fc_pred = compute_functional_connectivity(pred)
    fc_tgt = compute_functional_connectivity(target)
    mask = np.triu(np.ones_like(fc_pred, dtype=bool), k=1)
    a, b = fc_pred[mask], fc_tgt[mask]
    if len(a) > 1 and np.std(a) > 1e-10 and np.std(b) > 1e-10:
        r, _ = pearsonr(a, b)
        return float(r) if not np.isnan(r) else 0.0
    return 0.0


def plot_fc_gt_vs_pred(
    pred_full: np.ndarray,
    tgt_full: np.ndarray,
    save_path: str,
    title_prefix: str = "",
):
    """
    Save 1x2 figure: Ground Truth FC and Predicted FC (same color scale).
    pred_full, tgt_full: (T, V).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fc_pred = compute_functional_connectivity(pred_full)
    fc_tgt = compute_functional_connectivity(tgt_full)
    fc_sim = compute_fc_similarity(pred_full, tgt_full)

    vmin = float(min(fc_pred.min(), fc_tgt.min()))
    vmax = float(max(fc_pred.max(), fc_tgt.max()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axes[0].imshow(fc_tgt, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
    axes[0].set_title("Ground Truth FC")
    axes[0].set_xlabel("ROI")
    axes[0].set_ylabel("ROI")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(fc_pred, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
    axes[1].set_title("Predicted FC")
    axes[1].set_xlabel("ROI")
    axes[1].set_ylabel("ROI")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(f"{title_prefix}FC comparison | FC sim = {fc_sim:.4f}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_psd_spectrum_difference(
    pred_full: np.ndarray,
    tgt_full: np.ndarray,
    save_path: str,
    fs: float = 0.72,
):
    """
    Save 2x1 figure: average PSD (pred vs GT) and PSD difference.
    pred_full, tgt_full: (T, V).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T, V = pred_full.shape
    psd_pred_list, psd_tgt_list = [], []
    freqs = None

    for v in range(V):
        x, y = pred_full[:, v], tgt_full[:, v]
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
        return

    psd_pred_avg = np.mean(np.stack(psd_pred_list, axis=0), axis=0)
    psd_tgt_avg = np.mean(np.stack(psd_tgt_list, axis=0), axis=0)
    psd_diff = psd_pred_avg - psd_tgt_avg
    psd_mse = float(np.mean((psd_pred_avg - psd_tgt_avg) ** 2))
    psd_mae = float(np.mean(np.abs(psd_diff)))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(freqs, psd_tgt_avg, linewidth=2, label="GT")
    axes[0].plot(freqs, psd_pred_avg, linewidth=2, label="Predicted")
    axes[0].set_title("Average PSD (across ROIs)")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Power")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(freqs, psd_diff, linewidth=2)
    axes[1].axhline(0.0, linestyle="--", linewidth=1, alpha=0.7)
    axes[1].set_title(f"PSD Difference (Pred - GT) | PSD MSE={psd_mse:.6f}, PSD MAE={psd_mae:.6f}")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Δ Power")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_closest_subject_visualizations(
    per_subj: dict,
    subj_pred_chunks: dict,
    subj_tgt_chunks: dict,
    subj_starts: dict,
    subj_total_len: dict,
    aggregate_fn,
    out_dir: str,
    model_name: str = "model",
    fs: float = 0.72,
):
    """
    Find the closest subject by MSE, aggregate their pred/tgt timelines,
    then save FC and PSD plots.

    per_subj: dict[sid] = {"mse", "mae", "freq_diff", "fc_similarity"}
    subj_*: from evaluation loop (lists of chunks per subject)
    aggregate_fn: function(chunks, starts, total_len) -> (T, V) array
    """
    if not per_subj or out_dir is None:
        return
    best_sid = min(per_subj.keys(), key=lambda s: per_subj[s]["mse"])
    total_len = subj_total_len[best_sid]
    pred_full = aggregate_fn(subj_pred_chunks[best_sid], subj_starts[best_sid], total_len)
    tgt_full = aggregate_fn(subj_tgt_chunks[best_sid], subj_starts[best_sid], total_len)

    os.makedirs(out_dir, exist_ok=True)
    fc_path = os.path.join(out_dir, f"closest_subject_{best_sid}_fc.png")
    psd_path = os.path.join(out_dir, f"closest_subject_{best_sid}_psd.png")

    plot_fc_gt_vs_pred(
        pred_full, tgt_full, fc_path,
        title_prefix=f"{model_name} closest subject {best_sid} | ",
    )
    plot_psd_spectrum_difference(pred_full, tgt_full, psd_path, fs=fs)

    mse = per_subj[best_sid]["mse"]
    fc_sim = per_subj[best_sid]["fc_similarity"]
    print(f"[Viz] Closest subject {best_sid}  MSE={mse:.6f}  FC sim={fc_sim:.4f}")
    print(f"[Viz] Saved  {fc_path}")
    print(f"[Viz] Saved  {psd_path}")
