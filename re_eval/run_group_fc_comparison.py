#!/usr/bin/env python3
"""
Group-level Functional Connectivity (FC) comparison: real vs FM-fMRI generated vs baseline generated.

Loads the best FM-fMRI checkpoint and the best baseline (for a given task) by reading test_results.txt
in each load_dir and selecting the entry with the best FC top-5% precision (then FC similarity).
Runs both on the same test set and plots group-level average FC matrices:
  - Real (group average FC over test windows)
  - FM-fMRI generated (group average FC)
  - Baseline generated (group average FC)

Checkpoint directories are read from fm_fmri.json and fm_baseline_only.json (load_dir per entry).

Usage (from repo root):
  # Single task:
  python re_eval/run_group_fc_comparison.py --task_name emotion --save_dir re_eval/group_fc_plots

  # All tasks (separate figure per task):
  python re_eval/run_group_fc_comparison.py --all_tasks --save_dir re_eval/group_fc_plots

  # One figure: rows = tasks, columns = models (Real, FM-fMRI, baselines):
  python re_eval/run_group_fc_comparison.py --all_tasks_one_figure --save_dir re_eval/group_fc_plots

  # Optional: --fm_fmri_config, --baseline_config, --use_evs, --save_npy
"""

import os
import re
import sys
import json
import argparse
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
FM_FMRI_DIR = REPO_ROOT / "fm-fmri"
sys.path.insert(0, str(FM_FMRI_DIR))

# Re-use test loader and FM-fMRI loading from discriminative score
from run_discriminative_score import (
    get_test_loader_only,
    load_fmts_and_dataset,
    collect_real_and_generated,
)
from fc_utils import correlation_matrix


def parse_test_results_from_dir(load_dir):
    """
    Parse test_results.txt in load_dir for FC similarity and FC top-5% precision.
    Returns dict with keys fc_precision_at_5, fc_similarity (float or None if missing).
    """
    path = Path(load_dir) / "test_results.txt"
    out = {"fc_precision_at_5": None, "fc_similarity": None}
    if not path.exists():
        return out
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return out
    # FC similarity
    m = re.search(
        r"(?:FC sim(?:ilarity)?|Functional Connectivity Similarity).*?([0-9.e+-]+)\s*±",
        text,
        re.IGNORECASE,
    )
    if m:
        try:
            out["fc_similarity"] = float(m.group(1).strip())
        except ValueError:
            pass
    # k=5% precision
    m = re.search(
        r"k=5%\s*:\s*Precision\s+([0-9.e+-]+|nan)\s*±",
        text,
        re.IGNORECASE,
    )
    if m:
        val = m.group(1).strip().lower()
        if val != "nan":
            try:
                out["fc_precision_at_5"] = float(val)
            except ValueError:
                pass
    return out


def select_best_by_fc_precision(entries):
    """
    From a list of config entries (each with load_dir, name, ...), select the one with the
    best FC top-5% precision (from load_dir/test_results.txt). Tiebreak by FC similarity.
    Returns (best_entry, score_dict). If no entry has parseable metrics, returns (entries[0], {}).
    """
    if not entries:
        return None, {}
    scored = []
    for e in entries:
        load_dir = e.get("load_dir")
        if not load_dir:
            scored.append((e, float("-inf"), float("-inf")))
            continue
        meta = parse_test_results_from_dir(load_dir)
        prec = meta["fc_precision_at_5"]
        sim = meta["fc_similarity"]
        if prec is None:
            prec = float("-inf")
        if sim is None:
            sim = float("-inf")
        scored.append((e, prec, sim))
    # Sort by fc_precision_at_5 desc, then fc_similarity desc
    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    best = scored[0][0]
    score_dict = {"fc_precision_at_5": scored[0][1] if scored[0][1] != float("-inf") else None,
                  "fc_similarity": scored[0][2] if scored[0][2] != float("-inf") else None}
    return best, score_dict


def load_timevae_and_collect_generated(load_dir, test_loader, device, prediction_length):
    """
    Load TimeVAE from load_dir/best_model.pth and run on test_loader.
    Returns X_gen (N, T, V) numpy array. Uses same batch order as test_loader.
    """
    sys.path.insert(0, str(REPO_ROOT / "baselines"))
    import timevae_baseline as tv
    TimeVAE = tv.TimeVAE

    best_path = Path(load_dir) / "best_model.pth"
    if not best_path.exists():
        raise FileNotFoundError(f"TimeVAE checkpoint not found: {best_path}")

    ckpt = torch.load(best_path, map_location=device)
    saved_args = ckpt.get("args", {})
    if not isinstance(saved_args, dict):
        saved_args = vars(saved_args) if saved_args else {}
    num_variables = ckpt.get("num_variables", saved_args.get("num_variables", 166))
    hidden_dim = saved_args.get("hidden_dim", 128)
    num_layers = saved_args.get("num_layers", 2)
    latent_dim = saved_args.get("latent_dim", 64)
    dropout = saved_args.get("dropout", 0.1)
    max_pred_len = max(prediction_length, 256)

    model = TimeVAE(
        input_dim=num_variables,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        latent_dim=latent_dim,
        output_dim=num_variables,
        max_prediction_length=max_pred_len,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    gen_list = []
    with torch.no_grad():
        for batch in test_loader:
            input_seq = batch["input"].to(device).float()
            pred, _, _ = model(input_seq, prediction_length=prediction_length, target=None, teacher_forcing_ratio=0.0)
            if pred.dim() == 2:
                pred = pred.unsqueeze(1)
            gen_list.append(pred.cpu().numpy())
    X_gen = np.concatenate(gen_list, axis=0)
    return X_gen


def group_average_fc(X):
    """
    X: (N, T, V). Compute FC matrix per window, then average.
    Returns (V, V) group-level average FC.
    """
    N = X.shape[0]
    fc_list = [correlation_matrix(X[i]) for i in range(N)]
    return np.mean(fc_list, axis=0)


def plot_group_fc_comparison(fc_list, labels, save_path, title_suffix=""):
    """Plot 1 x n heatmaps: Real | FM-fMRI | Baseline1 | Baseline2 | ... Shared color scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    all_min = min(fc.min() for fc in fc_list)
    all_max = max(fc.max() for fc in fc_list)
    lim = max(abs(all_min), abs(all_max), 0.01)
    vmin, vmax = -lim, lim

    n = len(fc_list)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]
    im = None
    for ax, fc, label in zip(axes, fc_list, labels):
        im = ax.imshow(fc, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("ROI")
        ax.set_ylabel("ROI")
    # Single shared colorbar: standalone axis on the right of all subplots
    if im is not None:
        # Leave room for the colorbar on the right
        plt.tight_layout(rect=[0, 0, 0.9, 0.92])
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="Correlation")
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.suptitle(f"Group-level average Functional Connectivity{title_suffix}", fontsize=14, fontweight="bold")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# Column order for grid: same across all tasks
MODEL_COLUMN_ORDER = ["Real", "FM-fMRI", "timegan", "timevae", "ddpm", "diffusion_ts", "lstm_gan"]


def plot_group_fc_grid(task_names, grid_fcs, column_labels, save_path):
    """
    Plot one figure: rows = tasks, columns = models. Each cell is (V,V) FC heatmap.
    grid_fcs: list of length n_tasks; each element is list of length n_models of (V,V) arrays or None.
    column_labels: list of length n_models. Shared color scale across all cells.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    n_rows = len(task_names)
    n_cols = len(column_labels)
    # Flatten to get global vmin/vmax (skip None)
    all_fcs = []
    for row in grid_fcs:
        for fc in row:
            if fc is not None:
                all_fcs.append(fc)
    if not all_fcs:
        raise ValueError("No FC matrices to plot")
    all_min = min(fc.min() for fc in all_fcs)
    all_max = max(fc.max() for fc in all_fcs)
    lim = max(abs(all_min), abs(all_max), 0.01)
    vmin, vmax = -lim, lim

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]
    im = None
    for r, task in enumerate(task_names):
        for c in range(n_cols):
            ax = axes[r][c]
            fc = grid_fcs[r][c]
            if fc is not None:
                im = ax.imshow(fc, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
            else:
                ax.set_facecolor("#e0e0e0")
                ax.text(0.5, 0.5, "—", ha="center", va="center", fontsize=14)
            if r == 0:
                ax.set_title(column_labels[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(task, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Group-level average FC: rows = tasks, columns = models", fontsize=12, fontweight="bold")
    # One shared colorbar: standalone axis on the right of all subplots
    if im is not None:
        plt.tight_layout(rect=[0, 0, 0.9, 0.96])
        cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
        fig.colorbar(im, cax=cbar_ax, label="Correlation")
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    p = argparse.ArgumentParser(
        description="Group-level FC comparison: real vs FM-fMRI vs baseline (checkpoints from JSON configs)"
    )
    p.add_argument("--task_name", type=str, default=None,
                   help="Task name (e.g. emotion, WM). Required unless --all_tasks.")
    p.add_argument("--all_tasks", action="store_true",
                   help="Run group FC comparison for every task in the config (emotion, gambling, ...)")
    p.add_argument("--all_tasks_one_figure", action="store_true",
                   help="One figure: rows = tasks, columns = models (Real, FM-fMRI, baselines). Implies all tasks.")
    p.add_argument("--fm_fmri_config", type=str, default=None,
                   help="JSON config for FM-fMRI checkpoints (default: re_eval/re_eval/fm_fmri.json)")
    p.add_argument("--baseline_config", type=str, default=None,
                   help="JSON config for baselines (default: re_eval/re_eval/fm_baseline_only.json)")
    p.add_argument("--baseline_model_type", type=str, default=None,
                   help="Restrict baseline to this model_type (e.g. timevae). If None, use any.")
    p.add_argument("--data_root", type=str, default=os.getenv("DATA_ROOT", ""),
                   help="HCP rest data root")
    p.add_argument("--task_root", type=str, default=os.getenv("TASK_ROOT", ""),
                   help="HCP task data root")
    p.add_argument("--save_dir", type=str, default="re_eval/group_fc_plots",
                   help="Directory to save figure and optional .npy matrices")
    p.add_argument("--ode_steps", type=int, default=50, help="ODE steps for FM-fMRI sampling")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--lookback_length", type=int, default=512)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--use_evs", action="store_true")
    p.add_argument("--ev_root", type=str, default=None)
    p.add_argument("--save_npy", action="store_true", help="Save group FC matrices as .npy files")
    args = p.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Default config paths
    if not args.fm_fmri_config:
        args.fm_fmri_config = str(REPO_ROOT / "re_eval" / "re_eval" / "fm_fmri.json")
    if not args.baseline_config:
        args.baseline_config = str(REPO_ROOT / "re_eval" / "re_eval" / "fm_baseline_only.json")

    with open(args.fm_fmri_config, "r") as f:
        fm_config = json.load(f)
    if not isinstance(fm_config, list):
        fm_config = [fm_config]
    with open(args.baseline_config, "r") as f:
        bl_config = json.load(f)
    if not isinstance(bl_config, list):
        bl_config = [bl_config]

    task_names_all = sorted(set(e.get("task_name") for e in fm_config if e.get("task_name")))
    if not task_names_all:
        task_names_all = sorted(set(e.get("task_name") for e in bl_config if e.get("task_name")))
    if not task_names_all:
        raise SystemExit("No task_name found in configs")

    if args.all_tasks_one_figure:
        # One figure: rows = tasks, columns = models
        column_labels = ["Real", "FM-fMRI", "TimeGAN", "TimeVAE", "DDPM", "Diffusion-TS", "LSTM-GAN"]
        grid_rows = []  # (task_name, row of FCs)
        for task_name in task_names_all:
            print(f"Collecting data for task: {task_name}")
            try:
                fc_real, fc_fm, bl_by_type = _get_one_task_fc_data(args, task_name, fm_config, bl_config)
                row = [fc_real, fc_fm]
                for mt in MODEL_COLUMN_ORDER[2:]:
                    row.append(bl_by_type.get(mt))
                grid_rows.append((task_name, row))
            except Exception as e:
                print(f"  [Skip] {task_name}: {e}")
                if os.getenv("RE_EVAL_DEBUG"):
                    import traceback
                    traceback.print_exc()
        if not grid_rows:
            raise SystemExit("No task data collected for grid figure")
        task_names = [t for t, _ in grid_rows]
        grid_fcs = [row for _, row in grid_rows]
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = Path(args.save_dir).resolve() / "group_fc_comparison_all_tasks.pdf"
        plot_group_fc_grid(task_names, grid_fcs, column_labels, str(save_path))
        return

    if args.all_tasks:
        print(f"Running group FC comparison for {len(task_names_all)} tasks: {task_names_all}")
        for task_name in task_names_all:
            print("\n" + "=" * 60 + f" Task: {task_name} " + "=" * 60)
            try:
                _run_one_task(args, task_name, fm_config, bl_config)
            except Exception as e:
                print(f"  [FAIL] {task_name}: {e}")
                if os.getenv("RE_EVAL_DEBUG"):
                    import traceback
                    traceback.print_exc()
        return

    if not args.task_name:
        raise SystemExit("Either --task_name <name>, --all_tasks, or --all_tasks_one_figure is required")
    _run_one_task(args, args.task_name, fm_config, bl_config)


def _get_one_task_fc_data(args, task_name: str, fm_config: list, bl_config: list):
    """
    Load models and compute group-average FC for one task.
    Returns (fc_real, fc_fm, bl_by_type) where bl_by_type is dict model_type -> (V,V) FC.
    Raises on missing config; missing baselines are omitted from bl_by_type.
    """
    fm_entries = [e for e in fm_config if e.get("task_name") == task_name]
    if not fm_entries:
        raise SystemExit(f"No FM-fMRI entry with task_name={task_name} in config")
    fm_entry, _ = select_best_by_fc_precision(fm_entries)
    fm_load_dir = fm_entry["load_dir"]

    bl_entries_all = [e for e in bl_config if e.get("task_name") == task_name]
    if args.baseline_model_type:
        bl_entries_all = [e for e in bl_entries_all if e.get("model_type") == args.baseline_model_type]
    if not bl_entries_all:
        raise SystemExit(f"No baseline entry with task_name={task_name} in config")

    by_type = defaultdict(list)
    for e in bl_entries_all:
        by_type[e.get("model_type", "timevae")].append(e)
    bl_selected = []
    for mt in ["timegan", "timevae", "ddpm", "diffusion_ts", "lstm_gan"]:
        if mt not in by_type:
            continue
        best_entry, _ = select_best_by_fc_precision(by_type[mt])
        bl_selected.append((best_entry.get("name", mt), best_entry["load_dir"], mt))

    default_data = os.getenv("DATA_ROOT", "./data/hcp-resting-fc")
    default_task = os.getenv("TASK_ROOT", "./data/hcp-task-ts")
    ns = SimpleNamespace(
        data_root=args.data_root or default_data,
        task_root=args.task_root or default_task,
        task_name=task_name,
        use_evs=args.use_evs,
        ev_root=args.ev_root,
        lookback_length=args.lookback_length,
        prediction_length=None,
        stride=args.stride,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        load_dir=fm_load_dir,
        device=args.device,
        ode_steps=args.ode_steps,
    )
    for k in ["rest_hidden", "ctx_dim", "t_dim", "rest_encoder", "rest_patch_len", "rest_num_layers",
              "rest_nhead", "rest_dim_feedforward", "prior_K", "use_prior_detach", "num_conditions",
              "d_ev", "use_hrf_kernel", "hrf_kernel_len", "hrf_num_basis", "hrf_per_roi",
              "use_ev_hrf_timecourse", "ev_hrf_kernel_len", "ev_hrf_num_basis",
              "no_ev_hrf_delay_width", "ev_hrf_smooth_boxcar", "ev_hrf_boxcar_sigma"]:
        if not hasattr(ns, k):
            setattr(ns, k, None)

    test_loader, pred_len, V, _ = get_test_loader_only(ns)
    ns.prediction_length = pred_len
    model_fm, test_loader2, _, _, ns = load_fmts_and_dataset(ns)
    X_real, X_fm = collect_real_and_generated(
        model_fm, test_loader2, pred_len, args.device, ode_steps=args.ode_steps
    )
    n_real = X_real.shape[0]
    fc_real = group_average_fc(X_real[:n_real])
    fc_fm = group_average_fc(X_fm[:n_real])

    sys.path.insert(0, str(REPO_ROOT / "baselines"))
    try:
        from run_cfid_baseline import load_baseline_model, _collect_baseline_real_and_generated
    except ImportError:
        load_baseline_model = None
        _collect_baseline_real_and_generated = None

    bl_by_type = {}
    for bl_name, bl_load_dir, bl_model_type in bl_selected:
        if load_baseline_model is None:
            continue
        try:
            bl_load_path = Path(bl_load_dir).expanduser().resolve()
            model_bl = load_baseline_model(
                bl_load_path, bl_model_type, args.device, V, pred_len
            )
            _, X_gen = _collect_baseline_real_and_generated(
                model_bl, test_loader2, args.device, pred_len, bl_model_type,
                sample_steps=getattr(args, "ode_steps", 50),
            )
            n = min(n_real, X_gen.shape[0])
            fc_bl = group_average_fc(X_gen[:n])
            bl_by_type[bl_model_type] = fc_bl
        except Exception:
            pass
    return fc_real, fc_fm, bl_by_type


def _run_one_task(args, task_name: str, fm_config: list, bl_config: list):
    """Run group FC comparison for a single task. Uses task_name for filtering and output filenames."""
    print(f"Task: {task_name}")
    fm_entries = [e for e in fm_config if e.get("task_name") == task_name]
    if not fm_entries:
        raise SystemExit(f"No FM-fMRI entry with task_name={task_name} in config")
    fm_entry, fm_scores = select_best_by_fc_precision(fm_entries)
    fm_load_dir = fm_entry["load_dir"]
    if fm_scores.get("fc_precision_at_5") is not None or fm_scores.get("fc_similarity") is not None:
        print(f"FM-fMRI: prec@5={fm_scores.get('fc_precision_at_5')}, fc_sim={fm_scores.get('fc_similarity')}")
    bl_entries_all = [e for e in bl_config if e.get("task_name") == task_name]
    if args.baseline_model_type:
        bl_entries_all = [e for e in bl_entries_all if e.get("model_type") == args.baseline_model_type]
    if not bl_entries_all:
        raise SystemExit(f"No baseline entry with task_name={task_name} in config")
    by_type = defaultdict(list)
    for e in bl_entries_all:
        by_type[e.get("model_type", "timevae")].append(e)
    bl_selected = []
    for mt in ["timegan", "timevae", "ddpm", "diffusion_ts", "lstm_gan"]:
        if mt not in by_type:
            continue
        best_entry, _ = select_best_by_fc_precision(by_type[mt])
        bl_selected.append((best_entry.get("name", mt), best_entry["load_dir"], mt))
    print("Building test loader...")
    fc_real, fc_fm, bl_by_type = _get_one_task_fc_data(args, task_name, fm_config, bl_config)
    print(f"  Real/FM shapes collected; {len(bl_by_type)} baselines")

    # Single-task figure: same column order as grid
    fc_list = [fc_real, fc_fm]
    labels = ["Real (group avg)", "FM-fMRI (group avg)"]
    for mt in MODEL_COLUMN_ORDER[2:]:  # baselines only
        if mt in bl_by_type:
            fc_list.append(bl_by_type[mt])
            labels.append(f"{mt} (group avg)")
    baseline_fcs = [(mt, bl_by_type[mt]) for mt in MODEL_COLUMN_ORDER[2:] if mt in bl_by_type]

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = Path(args.save_dir).resolve() / f"group_fc_comparison_{task_name}.png"
    plot_group_fc_comparison(
        fc_list, labels, str(save_path),
        title_suffix=f" — {task_name}",
    )
    print(f"Figure saved to: {save_path}")

    if args.save_npy:
        np.save(Path(args.save_dir) / f"group_fc_real_{task_name}.npy", fc_real)
        np.save(Path(args.save_dir) / f"group_fc_fm_{task_name}.npy", fc_fm)
        for mt, fc_bl in baseline_fcs:
            np.save(Path(args.save_dir) / f"group_fc_baseline_{mt}_{task_name}.npy", fc_bl)
        print(f"Saved .npy matrices to {args.save_dir}")


if __name__ == "__main__":
    main()
