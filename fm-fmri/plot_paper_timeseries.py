#!/usr/bin/env python3
"""
Generate fMRI time series figure for paper: ground truth vs FM-generated.

Use this to create a publication-ready time series panel (one component of
your research figure). You can either:
  1. Load checkpoint + data and run one test batch (no .npy needed).
  2. Load saved arrays from a previous run (pred_full, tgt_full).
  3. Use a run directory that already contains pred_full.npy and tgt_full.npy.

Arrays: pred_full and tgt_full should be shape (T, V) = (timepoints, ROIs).

Usage:
  # Load checkpoint and data, run one test sample, then plot (no .npy needed)
  python fm-fmri/plot_paper_timeseries.py --checkpoint_dir path/to/checkpoints_fmts --out paper_timeseries.pdf

  # Override data paths if they differ from checkpoint
  python fm-fmri/plot_paper_timeseries.py --checkpoint_dir path/to/ckpt --data_root /path/to/rest --task_root /path/to/task --out fig.pdf

  # From two .npy files (e.g. saved during eval)
  python fm-fmri/plot_paper_timeseries.py --pred path/to/pred.npy --target path/to/tgt.npy --out paper_timeseries.pdf

  # From a run directory that has pred_full.npy and tgt_full.npy
  python fm-fmri/plot_paper_timeseries.py --run_dir path/to/run --out paper_timeseries.pdf

  # Plot options: ROIs, TR for time axis
  python fm-fmri/plot_paper_timeseries.py --checkpoint_dir path/to/ckpt --out fig.pdf --rois 0 10 50 100 --tr 0.72

  # With --checkpoint_dir, also get a second figure: Rest BOLD | Real task BOLD | Predicted task BOLD (three panels)
  python fm-fmri/plot_paper_timeseries.py --checkpoint_dir path/to/ckpt --out fig.pdf --out_rest_task_pred fig_rest_task_pred.pdf
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Distinct colors per ROI (red, yellow, blue, teal, darker blue); cycle if more than 5
ROI_COLORS = ["#c0392b", "#f1c40f", "#2980b9", "#16a085", "#1a5276"]


def _roi_color(i: int) -> str:
    return ROI_COLORS[i % len(ROI_COLORS)]


def _stacked_offset(signals: list, gap_scale: float = 1.3) -> tuple:
    """Compute vertical gap so stacked traces don't overlap. signals: list of 1d arrays."""
    ranges = [float(np.ptp(s)) for s in signals if s.size > 0]
    gap = (max(ranges) * gap_scale) if ranges else 1.0
    return gap


def plot_paper_timeseries(
    pred_full: np.ndarray,
    tgt_full: np.ndarray,
    save_path: str,
    title: str = "fMRI time series: Ground truth vs FM-generated",
    num_rois: int = 5,
    roi_indices: list = None,
    tr_sec: float = None,
    figsize: tuple = None,
    dpi: int = 300,
):
    """
    Paper-ready time series: one panel with all ROIs in different colors, vertically stacked (no overlap).
    pred_full, tgt_full: (T, V). Each ROI: GT (solid) and predicted (dashed) in same color.
    """
    T, V = pred_full.shape
    if roi_indices is not None:
        roi_idx = np.asarray(roi_indices, dtype=int)
        roi_idx = roi_idx[(roi_idx >= 0) & (roi_idx < V)]
        if len(roi_idx) == 0:
            roi_idx = np.linspace(0, V - 1, min(num_rois, V), dtype=int)
    else:
        roi_idx = np.linspace(0, V - 1, min(num_rois, V), dtype=int)
    n_roi = len(roi_idx)

    time_axis = np.arange(T, dtype=float)
    if tr_sec is not None:
        time_axis = time_axis * tr_sec
        xlabel = "Time (s)"
    else:
        xlabel = "Time (TR)"

    all_signals = [tgt_full[:, v] for v in roi_idx] + [pred_full[:, v] for v in roi_idx]
    gap = _stacked_offset(all_signals)

    if figsize is None:
        figsize = (8, 5)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for i, v in enumerate(roi_idx):
        color = _roi_color(i)
        y_gt = tgt_full[:, v] + i * gap
        y_pred = pred_full[:, v] + i * gap
        ax.plot(time_axis, y_gt, "-", color=color, linewidth=1.5, alpha=0.9)
        ax.plot(time_axis, y_pred, "--", color=color, linewidth=1.5, alpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_rest_task_pred(
    rest_full: np.ndarray,
    tgt_full: np.ndarray,
    pred_full: np.ndarray,
    save_path: str,
    title: str = "Rest, real task, and predicted task BOLD",
    num_rois: int = 5,
    roi_indices: list = None,
    tr_sec: float = None,
    figsize: tuple = None,
    dpi: int = 300,
):
    """
    One figure with three panels: Rest BOLD, Real task BOLD, Predicted task BOLD.
    rest_full: (L, V) rest window; tgt_full, pred_full: (T, V).
    Each panel shows all ROIs in different colors, vertically stacked (no overlap).
    Layout: 1 row x 3 columns.
    """
    L, V = rest_full.shape
    T = tgt_full.shape[0]
    if roi_indices is not None:
        roi_idx = np.asarray(roi_indices, dtype=int)
        roi_idx = roi_idx[(roi_idx >= 0) & (roi_idx < V)]
        if len(roi_idx) == 0:
            roi_idx = np.linspace(0, V - 1, min(num_rois, V), dtype=int)
    else:
        roi_idx = np.linspace(0, V - 1, min(num_rois, V), dtype=int)
    n_roi = len(roi_idx)

    time_rest = np.arange(L, dtype=float)
    time_task = np.arange(T, dtype=float)
    if tr_sec is not None:
        time_rest = time_rest * tr_sec
        time_task = time_task * tr_sec
        xlabel = "Time (s)"
    else:
        xlabel = "Time (TR)"

    if figsize is None:
        figsize = (12, 5)
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    col_titles = ["Rest BOLD", "Real task BOLD", "Predicted task BOLD"]
    for col, (data, t_axis) in enumerate([
        (rest_full, time_rest),
        (tgt_full, time_task),
        (pred_full, time_task),
    ]):
        ax = axes[col]
        ax.set_title(col_titles[col], fontsize=11, fontweight="bold")
        signals = [data[:, roi_idx[i]] for i in range(n_roi)]
        gap = _stacked_offset(signals)
        for i, v in enumerate(roi_idx):
            y = data[:, v] + i * gap
            ax.plot(t_axis, y, "-", color=_roi_color(i), linewidth=1.2, alpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def load_and_run_one_batch(
    checkpoint_dir: str,
    device: str = None,
    ode_steps: int = 50,
    data_root: str = None,
    task_root: str = None,
    task_name: str = None,
):
    """
    Load fm-fmri checkpoint and test data, run one test batch.
    Returns (pred_full, tgt_full, rest_full): pred/tgt (T, V), rest_full (L, V) for the three-panel Rest|Task|Pred figure.
    Uses saved args in checkpoint for model and data; data_root/task_root/task_name override if provided.
    """
    import torch
    from torch.utils.data import DataLoader

    fm_fmri_dir = Path(__file__).resolve().parent
    if str(fm_fmri_dir) not in sys.path:
        sys.path.insert(0, str(fm_fmri_dir))
    from dataset import HCPRestingFCDataset
    from fm_fmri import FMRIWindowDataset, FMTS

    checkpoint_dir = Path(checkpoint_dir)
    best_path = checkpoint_dir / "best_fmts.pth"
    if not best_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {best_path}. Need best_fmts.pth in --checkpoint_dir.")
    ckpt = torch.load(best_path, map_location="cpu")
    saved = ckpt.get("args") or {}
    if not saved:
        raise ValueError("Checkpoint has no saved args; cannot build model and data.")

    # Data paths: override with CLI if provided
    data_root = data_root or saved.get("data_root")
    task_root = task_root or saved.get("task_root")
    task_name = task_name or saved.get("task_name", "emotion")
    if not data_root or not task_root:
        raise ValueError("Checkpoint args missing data_root/task_root. Pass --data_root and --task_root.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_evs = saved.get("use_evs", False)
    ev_root = saved.get("ev_root")

    ds = HCPRestingFCDataset(
        data_root=data_root,
        task_root=task_root,
        task_name=task_name,
        use_evs=use_evs,
        ev_root=ev_root,
    )
    if len(ds) == 0:
        raise ValueError("Dataset has 0 subjects. Check --data_root and --task_root.")

    lookback = saved.get("lookback_length", 512)
    pred_len = saved.get("prediction_length")
    if pred_len is None:
        task_ts = ds.load_task_subject(ds.subject_ids[0])
        pred_len = task_ts.shape[0] if task_ts.ndim == 1 else task_ts.shape[0]
    stride = saved.get("stride", 10)
    normalize = saved.get("normalize", True)
    train_ratio = saved.get("train_ratio", 0.7)
    val_ratio = saved.get("val_ratio", 0.15)

    train_ds = FMRIWindowDataset(
        dataset=ds,
        lookback_length=lookback,
        prediction_length=pred_len,
        stride=stride,
        normalize=normalize,
        split="train",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        use_task_target=True,
    )
    test_ds = FMRIWindowDataset(
        dataset=ds,
        lookback_length=lookback,
        prediction_length=pred_len,
        stride=stride,
        normalize=normalize,
        split="test",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        use_task_target=True,
    )
    if normalize and train_ds.rest_means is not None:
        test_ds.rest_means, test_ds.rest_stds = train_ds.rest_means, train_ds.rest_stds
        test_ds.task_means, test_ds.task_stds = train_ds.task_means, train_ds.task_stds

    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(test_loader))
    V = int(batch["input"].shape[-1])

    model = FMTS(
        v_dim=V,
        rest_hidden=saved.get("rest_hidden", 256),
        ctx_dim=saved.get("ctx_dim", 256),
        t_dim=saved.get("t_dim", 128),
        rest_encoder=saved.get("rest_encoder", "transformer"),
        rest_patch_len=saved.get("rest_patch_len", 16),
        rest_num_layers=saved.get("rest_num_layers", 2),
        rest_nhead=saved.get("rest_nhead", 4),
        rest_dim_feedforward=saved.get("rest_dim_feedforward", 512),
        use_evs=use_evs,
        num_conditions=saved.get("num_conditions", 32),
        d_ev=saved.get("d_ev", 64),
        use_hrf_kernel=saved.get("use_hrf_kernel", False),
        hrf_kernel_len=saved.get("hrf_kernel_len", 20),
        hrf_num_basis=saved.get("hrf_num_basis", 3),
        hrf_per_roi=saved.get("hrf_per_roi", False),
        use_ev_hrf_timecourse=saved.get("use_ev_hrf_timecourse", False),
        ev_hrf_kernel_len=saved.get("ev_hrf_kernel_len", 20),
        ev_hrf_num_basis=saved.get("ev_hrf_num_basis", 3),
        ev_hrf_use_delay_width=not saved.get("no_ev_hrf_delay_width", True),
        ev_hrf_smooth_boxcar=saved.get("ev_hrf_smooth_boxcar", False),
        ev_hrf_boxcar_sigma=saved.get("ev_hrf_boxcar_sigma", 0.5),
        prior_K=saved.get("prior_K", 8),
        use_prior_detach=saved.get("use_prior_detach", False),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    x_rest = batch["input"].to(device).float()
    x_task = batch["target"].to(device).float()
    ev = batch.get("ev")
    ev_mask = batch.get("ev_mask")
    starts = batch.get("task_start_idx")
    if ev is not None:
        ev = ev.to(device).float()
    if ev_mask is not None:
        ev_mask = ev_mask.to(device).float()
    task_start_idx = None
    if starts is not None:
        task_start_idx = starts.to(device) if isinstance(starts, torch.Tensor) else torch.tensor(starts, device=device, dtype=torch.float32)

    with torch.no_grad():
        x_pred = model.sample(
            x_rest,
            T_pred=pred_len,
            steps=ode_steps,
            ev=ev,
            ev_mask=ev_mask,
            task_start_idx=task_start_idx,
        )
    pred_full = x_pred[0].cpu().numpy()
    tgt_full = x_task[0].cpu().numpy()
    rest_full = x_rest[0].cpu().numpy()  # (L, V) for Rest | Task | Pred figure
    return pred_full, tgt_full, rest_full


def main():
    p = argparse.ArgumentParser(description="Plot fMRI time series (GT vs FM-generated) for paper figure")
    p.add_argument("--checkpoint_dir", type=str, default=None,
                   help="Load model from this dir (best_fmts.pth), run one test batch, and plot. No .npy needed.")
    p.add_argument("--data_root", type=str, default=None, help="Override data path when using --checkpoint_dir")
    p.add_argument("--task_root", type=str, default=None, help="Override task path when using --checkpoint_dir")
    p.add_argument("--task_name", type=str, default=None, help="Override task name when using --checkpoint_dir")
    p.add_argument("--ode_steps", type=int, default=50, help="ODE steps when using --checkpoint_dir")
    p.add_argument("--pred", type=str, default=None, help="Path to predicted time series .npy (T, V)")
    p.add_argument("--target", type=str, default=None, help="Path to target/GT time series .npy (T, V)")
    p.add_argument("--run_dir", type=str, default=None,
                   help="Run directory containing pred_full.npy and tgt_full.npy (optional)")
    p.add_argument("--out", type=str, default="paper_timeseries.pdf", help="Output figure path (.pdf or .png)")
    p.add_argument("--title", type=str, default="fMRI time series: Ground truth vs FM-generated")
    p.add_argument("--num_rois", type=int, default=5, help="Number of ROIs to plot if --rois not set")
    p.add_argument("--rois", type=int, nargs="+", default=None, help="Specific ROI indices to plot")
    p.add_argument("--tr", type=float, default=None, help="TR in seconds for x-axis (e.g. 0.72 for HCP)")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--out_rest_task_pred", type=str, default=None,
                   help="Also save Rest | Real task | Predicted three-panel figure to this path (only when using --checkpoint_dir)")
    p.add_argument("--title_rest_task_pred", type=str, default="Rest, real task, and predicted task BOLD")
    args = p.parse_args()

    if not HAS_MPL:
        raise RuntimeError("matplotlib is required. Install with: pip install matplotlib")

    pred_full = tgt_full = rest_full = None
    if args.checkpoint_dir:
        print("Loading checkpoint and data, running one test batch...")
        pred_full, tgt_full, rest_full = load_and_run_one_batch(
            args.checkpoint_dir,
            ode_steps=args.ode_steps,
            data_root=args.data_root,
            task_root=args.task_root,
            task_name=args.task_name,
        )
        print(f"Got pred/tgt shape: {pred_full.shape}, rest shape: {rest_full.shape}")
    elif args.run_dir:
        run = Path(args.run_dir)
        pred_path = run / "pred_full.npy"
        tgt_path = run / "tgt_full.npy"
        if pred_path.exists() and tgt_path.exists():
            pred_full = np.load(pred_path)
            tgt_full = np.load(tgt_path)
        else:
            print(f"Warning: {pred_path} or {tgt_path} not found. Use --pred and --target or --checkpoint_dir.")
    if pred_full is None and args.pred:
        pred_full = np.load(args.pred)
    if tgt_full is None and args.target:
        tgt_full = np.load(args.target)
    if pred_full is None or tgt_full is None:
        p.print_help()
        raise SystemExit(
            "Error: Provide one of: (1) --checkpoint_dir to load model and run inference, "
            "(2) --pred and --target .npy paths, or (3) --run_dir with pred_full.npy and tgt_full.npy."
        )
    if pred_full.shape != tgt_full.shape:
        raise ValueError(f"Shape mismatch: pred {pred_full.shape} vs target {tgt_full.shape}")

    plot_paper_timeseries(
        pred_full,
        tgt_full,
        args.out,
        title=args.title,
        num_rois=args.num_rois,
        roi_indices=args.rois,
        tr_sec=args.tr,
        dpi=args.dpi,
    )

    # Rest | Real task | Predicted three-panel figure (only when we have rest from checkpoint run)
    if rest_full is not None:
        out_rest_task_pred = args.out_rest_task_pred
        if not out_rest_task_pred:
            stem = Path(args.out).stem
            ext = Path(args.out).suffix
            out_rest_task_pred = str(Path(args.out).parent / f"{stem}_rest_task_pred{ext}")
        plot_rest_task_pred(
            rest_full,
            tgt_full,
            pred_full,
            out_rest_task_pred,
            title=args.title_rest_task_pred,
            num_rois=args.num_rois,
            roi_indices=args.rois,
            tr_sec=args.tr,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
