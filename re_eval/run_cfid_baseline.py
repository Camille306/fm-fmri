#!/usr/bin/env python3
"""
Compute cFID-FC for a baseline model (timegan, timevae, ddpm, diffusion_ts) and write
load_dir/cfid_fc.txt. Uses the same HCP test loader as FM-TS re-eval.

Usage (from repo root):
  python re_eval/run_cfid_baseline.py --load_dir results/timegan/emotion --model_type timegan --task_name emotion
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
RE_EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "baselines"))
sys.path.insert(0, str(RE_EVAL_DIR))

# Re-eval imports (get_test_loader_only, cfid_fc)
from run_discriminative_score import get_test_loader_only
from fc_utils import cfid_fc


def _collect_baseline_real_and_generated(model, test_loader, device, pred_len, model_type, sample_steps=50):
    """Run baseline model on test_loader; return (X_real, X_gen) as (N,T,V) numpy arrays."""
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


def load_baseline_model(load_dir: Path, model_type: str, device: torch.device, V: int, pred_len: int):
    """Load baseline checkpoint and return model."""
    load_dir = Path(load_dir)
    if model_type == "timegan":
        from timegan_model import TimeGAN
        ckpt_path = load_dir / "best_timegan_fmts_eval.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)
        saved = ckpt.get("args") or {}
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
        ckpt_path = load_dir / "best_model.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)
        saved = ckpt.get("args") or {}
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
        ckpt_path = load_dir / "best_ddpm.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)
        saved = ckpt.get("args") or {}
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
        ckpt_path = load_dir / "best_diffusion_ts.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)
        saved = ckpt.get("args") or {}
        # DiffusionTS only accepts v_dim, num_timesteps, rest_hidden, ctx_dim, t_dim (no beta_start/beta_end)
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
        ckpt_path = load_dir / "best_lstm_gan_fmts_eval.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)
        saved = ckpt.get("args") or {}
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


def compute_cfid_baseline(
    load_dir,
    model_type: str,
    task_name: str,
    data_root: str,
    task_root: str,
    device=None,
    max_fc_dim: int = 500,
    seed: int = 42,
    sample_steps: int = 50,
    write_file: bool = True,
):
    """
    Compute cFID-FC for a baseline checkpoint in-process. Returns float or None on failure.
    If write_file is True, writes load_dir/cfid_fc.txt.
    """
    load_dir = Path(load_dir)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args_ns = argparse.Namespace(
        data_root=data_root,
        task_root=task_root,
        task_name=task_name,
        use_evs=False,
        ev_root=None,
        batch_size=16,
        device=str(device),
        train_ratio=0.7,
        val_ratio=0.15,
        lookback_length=512,
        prediction_length=None,
        stride=10,
    )
    try:
        test_loader, pred_len, V, _ = get_test_loader_only(args_ns)
        model = load_baseline_model(load_dir, model_type, device, V, pred_len)
        X_real, X_gen = _collect_baseline_real_and_generated(
            model, test_loader, device, pred_len, model_type, sample_steps=sample_steps
        )
        mfd = max_fc_dim if max_fc_dim > 0 else None
        rng = np.random.default_rng(seed)
        c = cfid_fc(X_real, X_gen, max_fc_dim=mfd, rng=rng)
        if write_file:
            out_path = load_dir / "cfid_fc.txt"
            with open(out_path, "w") as f:
                f.write(f"cfid_fc\t{c:.6f}\n")
        return float(c)
    except Exception as e:
        import traceback
        print(f"  [cFID-FC] {model_type} failed: {e}", flush=True)
        if os.getenv("RE_EVAL_DEBUG"):
            traceback.print_exc()
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--load_dir", type=str, required=True, help="Directory containing baseline checkpoint")
    p.add_argument("--model_type", type=str, required=True, choices=["timegan", "timevae", "ddpm", "diffusion_ts", "lstm_gan"])
    p.add_argument("--task_name", type=str, default="emotion")
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--task_root", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max_fc_dim", type=int, default=500, help="0 = full FC dimension")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample_steps", type=int, default=50, help="DDPM/Diffusion-TS sampling steps")
    args = p.parse_args()

    os.chdir(REPO_ROOT)
    data_root = args.data_root or os.getenv("DATA_ROOT", "./data/hcp-resting-fc")
    task_root = args.task_root or os.getenv("TASK_ROOT", "./data/hcp-task-ts")
    c = compute_cfid_baseline(
        args.load_dir, args.model_type, args.task_name, data_root, task_root,
        device=args.device, max_fc_dim=args.max_fc_dim, seed=args.seed,
        sample_steps=args.sample_steps, write_file=True,
    )
    if c is not None:
        print(f"cFID-FC: {c:.6f}  written to {Path(args.load_dir) / 'cfid_fc.txt'}")
    else:
        print("cFID-FC: failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
