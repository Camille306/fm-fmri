#!/usr/bin/env python3
"""
Re-evaluate Biopoint FM-fMRI (Flow Matching) runs: for each biopoint FM entry in config,
run biopoint/run_flow_matching_biopoint.py --eval_only with that load_dir, then parse
test_results.txt and collect into one TSV. Optionally compute cFID-FC in-process for each entry.

Expects config with model_type "fmts" and load_dir containing best_fmts.pth
(e.g. from build_biopoint_config.py: biopoint_fm/single, biopoint_fm/sweep/<name>, biopoint_fm/ablations/<name>).

Usage (from repo root):
  python re_eval/run_biopoint_fm_only_metrics.py \\
    --config re_eval/re_eval/biopoint.json \\
    --out_csv re_eval/results_biopoint_fm_only.tsv

  # Only collect from existing test_results.txt (do not re-run eval):
  python re_eval/run_biopoint_fm_only_metrics.py \\
    --config re_eval/re_eval/biopoint.json \\
    --out_csv re_eval/results_biopoint_fm_only.tsv --skip_run

  # Skip cFID-FC computation (faster; cfid_fc column will be empty):
  python re_eval/run_biopoint_fm_only_metrics.py ... --no_cfid
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BIOPOINT_FM_SCRIPT = REPO_ROOT / "biopoint" / "run_flow_matching_biopoint.py"
RE_EVAL_DIR = REPO_ROOT / "re_eval"


def parse_test_results(path: Path) -> dict:
    """Parse biopoint test_results.txt (MSE/MAE/PSD/FC sim; optional k=5% precision if written)."""
    out = {"mae": None, "psd": None, "fc_similarity": None, "fc_precision_at_5": None}
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8")
    for key, pattern in [
        ("mae", r"MAE\s*\([^)]*\):\s*([0-9.e+-]+)\s*±"),
        ("psd", r"PSD\s*\([^)]*\):\s*([0-9.e+-]+)\s*±"),
        ("fc_similarity", r"FC sim\s*\([^)]*\):\s*([0-9.e+-]+)\s*±"),
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                out[key] = float(m.group(1).strip())
            except ValueError:
                pass
    m = re.search(r"k=5%\s*:\s*Precision\s+([0-9.e+-]+|nan)\s*±", text, re.IGNORECASE)
    if m:
        val = m.group(1).strip().lower()
        if val != "nan":
            try:
                out["fc_precision_at_5"] = float(val)
            except ValueError:
                pass
    return out


def _setup_biopoint_fm_imports():
    """Ensure biopoint, fm-fmri, and re_eval can be imported."""
    for d in (REPO_ROOT, REPO_ROOT / "biopoint", REPO_ROOT / "fm-fmri", RE_EVAL_DIR):
        d = str(d)
        if d not in sys.path:
            sys.path.insert(0, d)


def compute_cfid_biopoint_fm(
    load_dir: str,
    data_root: str,
    csv_path: str,
    eprime_root: str | None = None,
    device: str = "cuda",
    max_fc_dim: int = 500,
    batch_size: int = 16,
    ode_steps: int = 50,
    seed: int = 42,
    compute_cfid: bool = True,
) -> tuple[float | None, float | None]:
    """
    Load Biopoint FM-TS checkpoint, run inference on test set, compute cFID-FC and FC precision@5%.
    Returns (cfid_val, fc_precision_at_5); either can be None on error. Requires biopoint and fm-fmri on path.
    """
    _setup_biopoint_fm_imports()
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from biopoint_dataset import BiopointDatasetAdapter
    from biopoint_window_dataset import BiopointWindowDataset
    from fm_fmri import FMTS

    from fc_utils import cfid_fc, compute_fc_precision_at_5_paired

    load_path = Path(load_dir) / "best_fmts.pth"
    if not load_path.exists():
        return None, None
    ckpt = torch.load(load_path, map_location="cpu", weights_only=True)
    saved_args = ckpt.get("args") or {}

    # Resolve eprime root
    if eprime_root is None:
        for cand in (REPO_ROOT / "biopoint" / "eprime_biopoint", REPO_ROOT / "biopoint" / "eprime_timing_download"):
            if cand.exists():
                eprime_root = str(cand)
                break
    if not eprime_root or not Path(eprime_root).exists():
        return None, None

    adapter = BiopointDatasetAdapter(
        data_root=data_root,
        csv_path=ckpt.get("csv_path") or csv_path,
        ev_root=eprime_root,
    )
    lookback = saved_args.get("lookback_length", 50)
    pred_len = saved_args.get("prediction_length", 100)
    use_evs = saved_args.get("use_evs", True)

    train_ds = BiopointWindowDataset(
        adapter,
        lookback_length=lookback,
        prediction_length=pred_len,
        use_evs=use_evs,
        normalize=True,
        split="train",
    )
    test_ds = BiopointWindowDataset(
        adapter,
        lookback_length=lookback,
        prediction_length=pred_len,
        use_evs=use_evs,
        normalize=True,
        split="test",
    )
    if train_ds.rest_means is not None:
        test_ds.rest_means = train_ds.rest_means
        test_ds.rest_stds = train_ds.rest_stds
        test_ds.task_means = train_ds.task_means
        test_ds.task_stds = train_ds.task_stds

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    # Infer V from first subject
    sid0 = adapter.subject_ids[0]
    V = adapter.load_subject(sid0).shape[1]
    v_dim = V
    rest_hidden = saved_args.get("rest_hidden", 256)
    ctx_dim = saved_args.get("ctx_dim", 256)
    t_dim = saved_args.get("t_dim", 128)
    num_conditions = saved_args.get("num_conditions", 32)
    d_ev = saved_args.get("d_ev", 64)

    model = FMTS(
        v_dim=v_dim,
        rest_hidden=rest_hidden,
        ctx_dim=ctx_dim,
        t_dim=t_dim,
        use_evs=use_evs,
        num_conditions=num_conditions,
        d_ev=d_ev,
    )
    state = ckpt.get("model")
    if state is not None:
        model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    rng = np.random.default_rng(seed)
    X_real_list, X_gen_list = [], []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["input"].to(device)
            y = batch["target"]
            ev = batch["ev"].to(device)
            ev_mask = batch["ev_mask"].to(device)
            task_start_idx = batch["task_start_idx"]
            if isinstance(task_start_idx, torch.Tensor):
                task_start_idx = task_start_idx.to(device)
            else:
                task_start_idx = torch.tensor(task_start_idx, device=device, dtype=torch.float32)
            y_hat = model.sample(
                x,
                T_pred=pred_len,
                steps=ode_steps,
                ev=ev,
                ev_mask=ev_mask,
                task_start_idx=task_start_idx,
            )
            if y_hat is None:
                continue
            y_hat = y_hat.cpu().numpy()
            y_np = y.numpy()
            if y_np.ndim == 2:
                y_np = y_np[:, :, None]
            if y_hat.ndim == 2:
                y_hat = y_hat[:, :, None]
            X_real_list.append(y_np)
            X_gen_list.append(y_hat)

    if not X_real_list or not X_gen_list:
        return None, None
    X_real = np.concatenate(X_real_list, axis=0)
    X_gen = np.concatenate(X_gen_list, axis=0)
    cfid_val = float(cfid_fc(X_real, X_gen, max_fc_dim=max_fc_dim, rng=rng)) if compute_cfid else None
    fc_precision_at_5 = compute_fc_precision_at_5_paired(X_real, X_gen)
    fc_prec5 = float(fc_precision_at_5) if not np.isnan(fc_precision_at_5) else None
    return cfid_val, fc_prec5


def run_fm_eval(load_dir: str, data_root: str, csv_path: str) -> bool:
    """Run biopoint/run_flow_matching_biopoint.py --eval_only --save_dir <load_dir>."""
    cmd = [
        sys.executable,
        str(BIOPOINT_FM_SCRIPT),
        "--eval_only",
        "--save_dir", load_dir,
        "--data_root", data_root,
        "--csv_path", csv_path,
    ]
    try:
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  [WARN] {BIOPOINT_FM_SCRIPT.name} exited {result.returncode}: {result.stderr[:500]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("  [WARN] Timeout")
        return False
    except Exception as e:
        print(f"  [WARN] {e}")
        return False


def main():
    p = argparse.ArgumentParser(description="Re-eval Biopoint FM-fMRI runs and collect metrics to TSV")
    p.add_argument("--config", type=str, default=None,
                   help="JSON config (default: re_eval/re_eval/biopoint.json)")
    p.add_argument("--out_csv", type=str, default=None, help="Output TSV path")
    p.add_argument("--data_root", type=str, default=None,
                   help="Biopoint data root (default: env DATA_ROOT or script default)")
    p.add_argument("--csv_path", type=str, default=None,
                   help="Biopoint csv_path (default: env or script default)")
    p.add_argument("--skip_run", action="store_true",
                   help="Only parse existing test_results.txt; do not run --eval_only")
    p.add_argument("--no_cfid", action="store_true",
                   help="Do not compute cFID-FC (faster; cfid_fc column left empty)")
    p.add_argument("--eprime_root", type=str, default=None,
                   help="Eprime/event timing root for Biopoint (default: biopoint/eprime_biopoint or eprime_timing_download)")
    args = p.parse_args()

    import os
    data_root = args.data_root or os.getenv("DATA_ROOT", "./data/biopoint_data")
    csv_path = args.csv_path or os.getenv("BIOPOINT_CSV", ".//fMRI_autism/v3_causality_lstm/causal_lstm_2025/biopoint_data.csv")

    config_path = args.config or str(REPO_ROOT / "re_eval" / "re_eval" / "biopoint.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    if not isinstance(config, list):
        config = [config]

    entries = [
        e for e in config
        if e.get("model_type") == "fmts"
        and (e.get("data_source") == "biopoint" or "biopoint" in e.get("name", ""))
    ]
    if not entries:
        print("No Biopoint FM (model_type=fmts) entries in config.")
        return

    print("Config:", config_path, "| Biopoint FM entries:", len(entries))
    if not args.skip_run:
        print("Running each Biopoint FM --eval_only...")
    col_order = ["name", "mae", "psd", "fc_similarity", "fc_precision_at_5", "cfid_fc", "discriminative_score"]
    rows = []

    for idx, entry in enumerate(entries):
        name = entry.get("name", f"biopoint_fm_{idx}")
        load_dir = entry.get("load_dir")
        if not load_dir:
            rows.append({c: name if c == "name" else "" for c in col_order})
            continue
        if not (Path(load_dir) / "best_fmts.pth").exists() and not args.skip_run:
            print(f"  [Skip] {name}: best_fmts.pth not found in {load_dir}")
            rows.append({c: name if c == "name" else "" for c in col_order})
            continue

        print(f"  [{idx+1}/{len(entries)}] {name} ...", flush=True)
        if not args.skip_run and BIOPOINT_FM_SCRIPT.exists():
            run_fm_eval(load_dir, data_root, csv_path)
        tr_path = Path(load_dir) / "test_results.txt"
        meta = parse_test_results(tr_path)
        cfid_val = ""
        fc_prec5_val = meta["fc_precision_at_5"]  # fallback from parsed file
        if (Path(load_dir) / "best_fmts.pth").exists():
            try:
                cfid_result, fc_prec5_computed = compute_cfid_biopoint_fm(
                    load_dir,
                    data_root,
                    csv_path,
                    eprime_root=args.eprime_root,
                    max_fc_dim=500,
                    compute_cfid=not args.no_cfid,
                )
                if cfid_result is not None:
                    cfid_val = f"{cfid_result:.6g}"
                if fc_prec5_computed is not None:
                    fc_prec5_val = fc_prec5_computed
            except Exception as e:
                if os.getenv("RE_EVAL_DEBUG"):
                    import traceback
                    traceback.print_exc()
                print(f"    [WARN] cFID/fc_prec@5: {e}")
        row = {
            "name": name,
            "mae": meta["mae"] if meta["mae"] is not None else "",
            "psd": meta["psd"] if meta["psd"] is not None else "",
            "fc_similarity": meta["fc_similarity"] if meta["fc_similarity"] is not None else "",
            "fc_precision_at_5": f"{fc_prec5_val:.6g}" if fc_prec5_val is not None else "",
            "cfid_fc": cfid_val,
            "discriminative_score": "",
        }
        rows.append(row)

    for r in rows:
        print("\t".join(str(r.get(c, "")) for c in col_order))
    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write("\t".join(col_order) + "\n")
            for r in rows:
                f.write("\t".join(str(r.get(c, "")) for c in col_order) + "\n")
        print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
