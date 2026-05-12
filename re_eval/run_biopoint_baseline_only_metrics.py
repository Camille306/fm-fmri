#!/usr/bin/env python3
"""
Run baseline-only evaluation for Biopoint: for each biopoint baseline entry in config,
run biopoint/run_baselines_biopoint.py --model X --eval_only, then parse test_results.txt
and collect into one TSV. Optionally compute cFID-FC in-process for each baseline.

Expects config with data_source=biopoint and model_type in timegan, timevae, diffusion_ts, ddpm, lstm_gan.
load_dir should be the directory containing the checkpoint (e.g. results_biopoint/timegan).

Why fc_precision_at_5 is often empty:
- fc_precision_at_5: Biopoint eval only writes MSE, MAE, PSD, FC sim. This script parses k=5% if present.

Usage (from repo root):
  python re_eval/run_biopoint_baseline_only_metrics.py \\
    --config re_eval/re_eval/biopoint.json \\
    --out_csv re_eval/results_biopoint_baseline_only.tsv

  # Skip cFID-FC (faster):
  python re_eval/run_biopoint_baseline_only_metrics.py ... --no_cfid
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RE_EVAL_DIR = REPO_ROOT / "re_eval"
BIOPOINT_SCRIPT = REPO_ROOT / "biopoint" / "run_baselines_biopoint.py"

BASELINE_MODELS = ["timegan", "timevae", "diffusion_ts", "ddpm", "lstm_gan"]

# Biopoint checkpoint filenames (run_baselines_biopoint.py)
BIOPOINT_CKPT_NAMES = {
    "timegan": "best_timegan.pth",
    "timevae": "best_model.pth",
    "diffusion_ts": "best_diffusion_ts.pth",
    "ddpm": "best_ddpm.pth",
    "lstm_gan": "best_lstm_gan.pth",
}


def parse_test_results(path: Path) -> dict:
    """Parse biopoint test_results.txt (MSE/MAE/PSD/FC sim; optional k=5% precision if written)."""
    out = {"mae": None, "psd": None, "fc_similarity": None, "fc_precision_at_5": None}
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8")
    # "MAE (mean ± std): 0.123 ± 0.045" -> 0.123
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
    # k=5% precision (Biopoint eval does not write this yet; parse if present)
    m = re.search(r"k=5%\s*:\s*Precision\s+([0-9.e+-]+|nan)\s*±", text, re.IGNORECASE)
    if m:
        val = m.group(1).strip().lower()
        if val != "nan":
            try:
                out["fc_precision_at_5"] = float(val)
            except ValueError:
                pass
    return out


def _setup_imports():
    """Ensure biopoint, baselines, and re_eval can be imported."""
    for d in (REPO_ROOT, REPO_ROOT / "biopoint", REPO_ROOT / "baselines", RE_EVAL_DIR):
        d = str(d)
        if d not in sys.path:
            sys.path.insert(0, d)


def _get_biopoint_saved_args(load_dir: Path, model_type: str):
    """Load Biopoint baseline checkpoint and return saved args dict."""
    import torch
    ckpt_name = BIOPOINT_CKPT_NAMES.get(model_type)
    if not ckpt_name:
        return {}
    ckpt_path = Path(load_dir) / ckpt_name
    if not ckpt_path.exists():
        return {}
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    return ckpt.get("args") or {}


def _build_biopoint_test_loader(data_root: str, csv_path: str, saved_args: dict, batch_size: int = 16):
    """Build Biopoint adapter, train/test window datasets, copy norm from train to test, return (test_loader, pred_len, V)."""
    from torch.utils.data import DataLoader
    from biopoint_dataset import BiopointDatasetAdapter
    from biopoint_window_dataset import BiopointWindowDataset

    adapter = BiopointDatasetAdapter(data_root=data_root, csv_path=csv_path)
    lookback = saved_args.get("lookback_length", 200)
    pred_len = saved_args.get("prediction_length")
    if pred_len is None:
        sid0 = adapter.subject_ids[0]
        pred_len = adapter.load_task_subject(sid0).shape[0]
    train_ratio = saved_args.get("train_ratio", 0.7)
    val_ratio = saved_args.get("val_ratio", 0.15)
    stride = saved_args.get("stride", 10)
    normalize = saved_args.get("normalize", True)

    train_ds = BiopointWindowDataset(
        adapter,
        lookback_length=lookback,
        prediction_length=pred_len,
        stride=stride,
        normalize=normalize,
        split="train",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    test_ds = BiopointWindowDataset(
        adapter,
        lookback_length=lookback,
        prediction_length=pred_len,
        stride=stride,
        normalize=normalize,
        split="test",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    if train_ds.rest_means is not None:
        test_ds.rest_means = train_ds.rest_means
        test_ds.rest_stds = train_ds.rest_stds
        test_ds.task_means = train_ds.task_means
        test_ds.task_stds = train_ds.task_stds

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    V = adapter.load_subject(adapter.subject_ids[0]).shape[1]
    return test_loader, pred_len, V


def _load_baseline_model_biopoint(load_dir: Path, model_type: str, device, V: int, pred_len: int):
    """Load baseline model from Biopoint checkpoint (same classes as run_cfid_baseline, different ckpt names)."""
    import torch
    load_dir = Path(load_dir)
    ckpt_name = BIOPOINT_CKPT_NAMES.get(model_type)
    if not ckpt_name:
        raise ValueError(f"Unknown model_type={model_type}")
    ckpt_path = load_dir / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    saved = ckpt.get("args") or {}

    if model_type == "timegan":
        from timegan_model import TimeGAN
        model = TimeGAN(
            input_dim=V,
            hidden_dim=saved.get("hidden_dim", 128),
            num_layers=saved.get("num_layers", 2),
            dropout=saved.get("dropout", 0.1),
            max_prediction_length=pred_len,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
    elif model_type == "timevae":
        from timevae_baseline import TimeVAE
        model = TimeVAE(
            input_dim=V,
            hidden_dim=saved.get("hidden_dim", 128),
            num_layers=saved.get("num_layers", 2),
            latent_dim=saved.get("latent_dim", 64),
            output_dim=V,
            max_prediction_length=max(pred_len, 256),
            dropout=saved.get("dropout", 0.1),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
    elif model_type == "ddpm":
        from ddpm_baseline import DDPM
        model = DDPM(
            v_dim=V,
            num_timesteps=saved.get("num_timesteps", 1000),
            rest_hidden=saved.get("rest_hidden", 256),
            ctx_dim=saved.get("ctx_dim", 256),
            beta_start=saved.get("beta_start", 1e-4),
            beta_end=saved.get("beta_end", 0.02),
        ).to(device)
        model.load_state_dict(ckpt["model"])
    elif model_type == "diffusion_ts":
        from diffusion_ts_baseline import DiffusionTS
        model = DiffusionTS(
            v_dim=V,
            num_timesteps=saved.get("num_timesteps", 1000),
            rest_hidden=saved.get("rest_hidden", 256),
            ctx_dim=saved.get("ctx_dim", 256),
            t_dim=saved.get("t_dim", 128),
        ).to(device)
        model.load_state_dict(ckpt["model"])
    elif model_type == "lstm_gan":
        from lstm_gan_model import LSTMGAN
        model = LSTMGAN(
            input_dim=V,
            hidden_dim=saved.get("hidden_dim", 128),
            num_layers=saved.get("num_layers", 2),
            dropout=saved.get("dropout", 0.1),
            max_prediction_length=pred_len,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        raise ValueError(f"Unknown model_type={model_type}")
    return model


def _collect_baseline_real_and_generated(model, test_loader, device, pred_len, model_type, sample_steps=50):
    """Run baseline on test_loader; return (X_real, X_gen) as (N,T,V) numpy arrays."""
    import numpy as np
    import torch
    model.eval()
    reals, gens = [], []
    with torch.no_grad():
        for batch in test_loader:
            x_rest = batch["input"].to(device).float()
            x_task = batch["target"].to(device).float()
            if model_type == "timegan":
                x_pred = model(x_rest, prediction_length=pred_len)
            elif model_type == "lstm_gan":
                x_pred = model(x_rest, prediction_length=pred_len)
            elif model_type == "timevae":
                pred, _, _ = model(x_rest, prediction_length=pred_len, target=None, teacher_forcing_ratio=0.0)
                x_pred = pred.unsqueeze(1) if pred.dim() == 2 else pred
            elif model_type in ("ddpm", "diffusion_ts"):
                x_pred = model.sample(x_rest, T_pred=pred_len, num_steps=sample_steps)
            else:
                raise ValueError(f"Unknown model_type={model_type}")
            if x_pred.dim() == 2:
                x_pred = x_pred.unsqueeze(1)
            reals.append(x_task.cpu().numpy())
            gens.append(x_pred.cpu().numpy())
    X_real = np.concatenate(reals, axis=0)
    X_gen = np.concatenate(gens, axis=0)
    return X_real, X_gen


def compute_cfid_biopoint_baseline(
    load_dir: str,
    model_type: str,
    data_root: str,
    csv_path: str,
    device: str = "cuda",
    max_fc_dim: int = 500,
    batch_size: int = 16,
    sample_steps: int = 50,
    seed: int = 42,
    compute_cfid: bool = True,
) -> tuple[float | None, float | None]:
    """Load Biopoint baseline checkpoint, run inference on test set, compute cFID-FC and FC precision@5%.
    Returns (cfid_val, fc_precision_at_5); either can be None on error."""
    _setup_imports()
    import numpy as np
    import torch
    from fc_utils import cfid_fc, compute_fc_precision_at_5_paired

    load_path = Path(load_dir)
    saved_args = _get_biopoint_saved_args(load_path, model_type)
    ckpt_name = BIOPOINT_CKPT_NAMES.get(model_type)
    if ckpt_name and not (load_path / ckpt_name).exists():
        return None, None
    try:
        test_loader, pred_len, V = _build_biopoint_test_loader(data_root, csv_path, saved_args, batch_size=batch_size)
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        model = _load_baseline_model_biopoint(load_path, model_type, dev, V, pred_len)
        X_real, X_gen = _collect_baseline_real_and_generated(
            model, test_loader, dev, pred_len, model_type, sample_steps=sample_steps
        )
        cfid_val = float(cfid_fc(X_real, X_gen, max_fc_dim=max_fc_dim, rng=np.random.default_rng(seed))) if compute_cfid else None
        fc_prec5 = compute_fc_precision_at_5_paired(X_real, X_gen)
        fc_prec5_val = float(fc_prec5) if not np.isnan(fc_prec5) else None
        return cfid_val, fc_prec5_val
    except Exception as e:
        if os.getenv("RE_EVAL_DEBUG"):
            import traceback
            traceback.print_exc()
        print(f"    [WARN] cFID/fc_prec@5: {e}")
        return None, None


def run_baseline_eval(model: str, load_dir: str, data_root: str, csv_path: str) -> bool:
    """Run biopoint/run_baselines_biopoint.py --model X --eval_only --save_dir <parent>.
    Script appends model name to save_dir, so we pass parent of load_dir and model name.
    """
    load_path = Path(load_dir)
    save_dir_arg = str(load_path.parent)
    cmd = [
        sys.executable,
        str(BIOPOINT_SCRIPT),
        "--model", model,
        "--eval_only",
        "--save_dir", save_dir_arg,
        "--data_root", data_root,
        "--csv_path", csv_path,
    ]
    try:
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  [WARN] {BIOPOINT_SCRIPT.name} exited {result.returncode}: {result.stderr[:500]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("  [WARN] Timeout")
        return False
    except Exception as e:
        print(f"  [WARN] {e}")
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None,
                    help="JSON config (default: re_eval/re_eval/biopoint.json)")
    p.add_argument("--out_csv", type=str, default=None, help="Output TSV path")
    p.add_argument("--data_root", type=str, default=None,
                    help="Biopoint data root (default: env DATA_ROOT or script default)")
    p.add_argument("--csv_path", type=str, default=None,
                    help="Biopoint csv_path (default: env or script default)")
    p.add_argument("--results_biopoint", type=str, default=None,
                    help="Override load_dir base: use results_biopoint/<model> (default: use config)")
    p.add_argument("--skip_run", action="store_true",
                    help="Only parse existing test_results.txt; do not run --eval_only")
    p.add_argument("--no_cfid", action="store_true",
                    help="Do not compute cFID-FC (faster; cfid_fc column left empty)")
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
        if e.get("model_type") in BASELINE_MODELS
        and (e.get("data_source") == "biopoint" or "biopoint" in e.get("name", ""))
    ]
    if not entries:
        print("No biopoint baseline entries in config (model_type in timegan/timevae/diffusion_ts/ddpm/lstm_gan).")
        return

    results_biopoint_override = Path(args.results_biopoint).resolve() if args.results_biopoint else None
    if results_biopoint_override:
        print("Results base (override):", results_biopoint_override)
    print("Config:", config_path, "| Biopoint baseline entries:", len(entries))
    if not args.skip_run:
        print("Running each biopoint baseline --eval_only...")
    col_order = ["name", "mae", "psd", "fc_similarity", "fc_precision_at_5", "cfid_fc", "discriminative_score"]
    rows = []

    for idx, entry in enumerate(entries):
        name = entry.get("name", f"biopoint_baseline_{idx}")
        load_dir = entry.get("load_dir")
        model_type = entry.get("model_type")
        if results_biopoint_override is not None:
            load_dir = str(results_biopoint_override / model_type)
        if not load_dir or model_type not in BASELINE_MODELS:
            continue
        if not BIOPOINT_SCRIPT.exists():
            print(f"[Skip] {name}: script not found {BIOPOINT_SCRIPT}")
            rows.append({c: name if c == "name" else "" for c in col_order})
            continue

        print(f"  [{idx+1}/{len(entries)}] {name} ...", flush=True)
        if not args.skip_run:
            run_baseline_eval(model_type, load_dir, data_root, csv_path)
        tr_path = Path(load_dir) / "test_results.txt"
        meta = parse_test_results(tr_path)
        cfid_val = ""
        fc_prec5_val = meta["fc_precision_at_5"]  # fallback from parsed file
        if model_type in BIOPOINT_CKPT_NAMES:
            ckpt_path = Path(load_dir) / BIOPOINT_CKPT_NAMES[model_type]
            if ckpt_path.exists():
                try:
                    cfid_result, fc_prec5_computed = compute_cfid_biopoint_baseline(
                        load_dir, model_type, data_root, csv_path,
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
