"""
Generate synthetic task time series from a trained fm-fmri (flow matching) Biopoint model.
Saves one (T_pred, V) array per subject to synthetic_dir for use with STAGIN real+synthetic training.
Run from repo root so fm-fmri and biopoint are importable.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Paths for fm-fmri and biopoint
_repo_root = Path(__file__).resolve().parent.parent
_fm_fmri_dir = _repo_root / "fm-fmri"
_biopoint_dir = _repo_root / "biopoint"
sys.path.insert(0, str(_fm_fmri_dir))
sys.path.insert(0, str(_biopoint_dir))

# EV: optional; if not available we pass zeros
MAX_EV_EVENTS = 64


def load_rest_window(data_root, subject_id, lookback_length, ts_filename_suffix):
    """Load rest time series and return one window (lookback_length, V)."""
    suffix = ts_filename_suffix if ts_filename_suffix.startswith("_") else f"_{ts_filename_suffix}"
    path = os.path.join(data_root, "output", subject_id, "rest", f"{subject_id}{suffix}")
    ts = np.load(path).astype(np.float32)
    if ts.ndim == 1:
        ts = ts.reshape(-1, 1)
    if ts.shape[0] < lookback_length:
        raise ValueError(f"Subject {subject_id} rest length {ts.shape[0]} < lookback_length {lookback_length}")
    return ts[:lookback_length]  # (lookback_length, V)


def load_ev_if_available(adapter, subject_id):
    """Return (ev, ev_mask) for the subject, or (zeros, zeros) if no EV."""
    if adapter is None or not getattr(adapter, "use_evs", False):
        return np.zeros((MAX_EV_EVENTS, 4), dtype=np.float32), np.zeros(MAX_EV_EVENTS, dtype=np.float32)
    try:
        ev_full = adapter.load_ev_subject(subject_id)
        N_ev, n_cols = ev_full.shape
        if N_ev > MAX_EV_EVENTS:
            ev_full = ev_full[:MAX_EV_EVENTS]
            N_ev = MAX_EV_EVENTS
        pad_len = MAX_EV_EVENTS - N_ev
        if pad_len > 0:
            ev_full = np.concatenate([ev_full, np.zeros((pad_len, n_cols), dtype=np.float32)], axis=0)
        ev_mask = np.zeros(MAX_EV_EVENTS, dtype=np.float32)
        ev_mask[:N_ev] = 1.0
        return ev_full.astype(np.float32), ev_mask
    except Exception:
        return np.zeros((MAX_EV_EVENTS, 4), dtype=np.float32), np.zeros(MAX_EV_EVENTS, dtype=np.float32)


def main():
    p = argparse.ArgumentParser(description="Generate synthetic Biopoint task from fm-fmri for STAGIN augmentation")
    p.add_argument("--data_root", type=str, default="./data/biopoint_data", help="Biopoint data root (output/<id>/rest, task) for shen268 mode")
    p.add_argument("--csv_path", type=str, default="./data/biopoint_data.csv", help="CSV with subject_id, group")
    p.add_argument("--atlas_source", type=str, default="shen268", choices=["shen268", "dk"],
                   help="Which atlas to use for loading the conditioning rest time series")
    p.add_argument("--dk_atlas_ts_root", type=str, default="./data/biopoint_dk_atlas",
                   help="Root containing <subject_id>_rest_roi_ts.pt and <subject_id>_task_roi_ts.pt (when atlas_source=dk)")
    p.add_argument("--ts_filename_suffix", type=str, default="_shen268_ts.npy",
                   help="ROI file suffix when atlas_source=shen268 (e.g. _aal_ts.npy or _shen268_ts.npy)")
    p.add_argument("--fm_checkpoint", type=str, required=True, help="Path to best_fmts.pth (from run_flow_matching_biopoint)")
    p.add_argument("--synthetic_dir", type=str, default="./synthetic_biopoint", help="Where to save *_syn.npy and synthetic_manifest.csv")
    p.add_argument("--eprime_root", type=str, default=None, help="If set, use EV for generation (BiopointDatasetAdapter)")
    p.add_argument("--ode_steps", type=int, default=50)
    p.add_argument("--override_prediction_length", type=int, default=None,
                   help="Override checkpoint prediction_length (e.g. 146 to match rest length)")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    csv_path = args.csv_path or os.path.join(args.data_root, "biopoint_data.csv")
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(args.data_root, csv_path)
    df = pd.read_csv(csv_path)
    subject_ids = df["subject_id"].astype(str).tolist()
    group_col = "group" if "group" in df.columns else df.columns[1]
    group_dict = {str(row["subject_id"]): row[group_col] for _, row in df.iterrows()}

    # Build adapter (also used for DK rest loading and/or EV filtering)
    adapter = None
    from biopoint_dataset import BiopointDatasetAdapter
    ev_root = args.eprime_root if args.eprime_root and os.path.isdir(args.eprime_root) else None
    adapter = BiopointDatasetAdapter(
        data_root=args.data_root,
        csv_path=csv_path,
        ev_root=ev_root,
        atlas_source=("dk" if args.atlas_source == "dk" else "shen268"),
        dk_atlas_ts_root=args.dk_atlas_ts_root,
    )
    if ev_root:
        print(f"Using EV from {ev_root}; {len(adapter.subject_ids)} subjects with EV")
    else:
        print(f"No/invalid eprime_root; generating without EV filtering; {len(adapter.subject_ids)} subjects with rest/task data")
    subject_ids = [s for s in subject_ids if s in adapter.subject_ids]
    print(f"Subjects kept after adapter filtering: {len(subject_ids)}")

    # Load fm-fmri checkpoint and model
    ckpt = torch.load(args.fm_checkpoint, map_location=device)
    saved_args = ckpt.get("args", {})
    lookback_length = saved_args.get("lookback_length", 200)
    prediction_length = saved_args.get("prediction_length", 116)
    V = saved_args.get("v_dim")
    if V is None:
        # Infer V from first subject
        try:
            if args.atlas_source == "dk":
                w = adapter.load_subject(subject_ids[0])[:lookback_length]
            else:
                w = load_rest_window(args.data_root, subject_ids[0], lookback_length, args.ts_filename_suffix)
            V = w.shape[1]
        except Exception as e:
            raise RuntimeError(f"Cannot infer V from data: {e}") from e
    if args.override_prediction_length is not None:
        print(f"Overriding prediction_length: {prediction_length} -> {args.override_prediction_length}")
        prediction_length = args.override_prediction_length
    print(f"FM checkpoint: lookback={lookback_length}, prediction_length={prediction_length}, V={V}")

    from fm_fmri import FMTS
    model = FMTS(
        v_dim=V,
        rest_hidden=saved_args.get("rest_hidden", 256),
        ctx_dim=saved_args.get("ctx_dim", 256),
        t_dim=saved_args.get("t_dim", 128),
        use_evs=bool(saved_args.get("use_evs", True)),
        num_conditions=saved_args.get("num_conditions", 32),
        d_ev=saved_args.get("d_ev", 64),
        rest_nhead=saved_args.get("rest_nhead", 4),
        rest_num_layers=saved_args.get("rest_num_layers", 2),
        prior_K=saved_args.get("prior_K", 8),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    os.makedirs(args.synthetic_dir, exist_ok=True)
    manifest_rows = []

    for subject_id in subject_ids:
        try:
            if args.atlas_source == "dk":
                rest_full = adapter.load_subject(subject_id)
                if rest_full.shape[0] < lookback_length:
                    raise ValueError(f"Subject {subject_id} rest length {rest_full.shape[0]} < lookback_length {lookback_length}")
                rest = rest_full[:lookback_length]
            else:
                rest = load_rest_window(args.data_root, subject_id, lookback_length, args.ts_filename_suffix)
        except Exception as e:
            print(f"Skip {subject_id}: {e}")
            continue
        ev_np, ev_mask_np = load_ev_if_available(adapter, subject_id)
        x_rest = torch.from_numpy(rest).float().unsqueeze(0).to(device)  # (1, L, V)
        ev = torch.from_numpy(ev_np).float().unsqueeze(0).to(device)
        ev_mask = torch.from_numpy(ev_mask_np).float().unsqueeze(0).to(device)
        task_start_idx = torch.zeros(1, device=device)

        with torch.no_grad():
            x_pred = model.sample(
                x_rest,
                T_pred=prediction_length,
                steps=args.ode_steps,
                ev=ev,
                ev_mask=ev_mask,
                task_start_idx=task_start_idx,
            )
        syn = x_pred.squeeze(0).cpu().numpy()  # (T_pred, V)
        out_path = os.path.join(args.synthetic_dir, f"{subject_id}_syn.npy")
        np.save(out_path, syn.astype(np.float32))
        group = group_dict.get(subject_id, "con")
        label = 1 if str(group).strip().lower() in ("pat", "patient", "asd", "1") else 0
        manifest_rows.append({"subject_id": subject_id, "label": label, "path": out_path})

    manifest_path = os.path.join(args.synthetic_dir, "synthetic_manifest.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"Saved {len(manifest_rows)} synthetic time series to {args.synthetic_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
