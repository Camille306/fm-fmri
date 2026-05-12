#!/usr/bin/env python3
"""
Train and evaluate baseline models (TimeGAN, TimeVAE, Diffusion-TS, DDPM) for
rest-to-task on Biopoint data. Uses Biopoint windowed dataset and the same
subject-level evaluation as the baselines folder.

Run from repo root:
  python biopoint/run_baselines_biopoint.py --model timegan --data_root /path/to/biopoint_data --save_dir ./results_biopoint/timegan
  python biopoint/run_baselines_biopoint.py --model timevae --data_root /path/to/biopoint_data --save_dir ./results_biopoint/timevae
  python biopoint/run_baselines_biopoint.py --model diffusion_ts --data_root /path/to/biopoint_data --save_dir ./results_biopoint/diffusion_ts
  python biopoint/run_baselines_biopoint.py --model ddpm --data_root /path/to/biopoint_data --save_dir ./results_biopoint/ddpm
"""

import os
import sys
import csv
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure biopoint is first so baseline "from dataset import HCPRestingFCDataset" resolves to biopoint/dataset.py
_biopoint_dir = Path(__file__).resolve().parent
_repo_root = _biopoint_dir.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_biopoint_dir))

from biopoint_dataset import BiopointDatasetAdapter
from biopoint_window_dataset import BiopointWindowDataset


def get_args():
    p = argparse.ArgumentParser(description="Train baselines on Biopoint rest-to-task")
    p.add_argument("--model", type=str, default="timegan", choices=["timegan", "timevae", "diffusion_ts", "ddpm", "lstm_gan"],
               help="Baseline model to train (default: timegan for quick testing)")
    p.add_argument("--data_root", type=str, default="./data/biopoint_data")
    p.add_argument("--csv_path", type=str, default="./data/biopoint_data.csv")
    p.add_argument("--atlas_source", type=str, default="dk", choices=["dk", "shen268"],
                   help="Which atlas ROI time-series to use (default: dk)")
    p.add_argument("--dk_atlas_ts_root", type=str, default="./data/biopoint_dk_atlas",
                   help="Root containing <subject_id>_rest_roi_ts.pt and <subject_id>_task_roi_ts.pt (when atlas_source=dk)")
    p.add_argument("--save_dir", type=str, default="./results_biopoint")
    p.add_argument("--lookback_length", type=int, default=200)
    p.add_argument("--prediction_length", type=int, default=None)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--normalize", action="store_true", default=True)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--eval_only", action="store_true", help="Load best checkpoint and run test eval + viz only")
    # TimeGAN-specific
    p.add_argument("--embedder_epochs", type=int, default=20)
    p.add_argument("--supervisor_epochs", type=int, default=10)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lambda_embed", type=float, default=10.0)
    p.add_argument("--lambda_supervise", type=float, default=1.0)
    p.add_argument("--lambda_adv", type=float, default=0.1)
    p.add_argument("--freq_loss_weight", type=float, default=0.0)
    # TimeVAE-specific
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--beta_warmup_epochs", type=int, default=10)
    # Diffusion/DDPM
    p.add_argument("--rest_hidden", type=int, default=256)
    p.add_argument("--ctx_dim", type=int, default=256)
    p.add_argument("--num_timesteps", type=int, default=1000)
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--freq_aux_steps", type=int, default=10)
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=0.02)
    return p.parse_args()


def build_dataloaders(args):
    adapter = BiopointDatasetAdapter(
        data_root=args.data_root,
        csv_path=args.csv_path,
        atlas_source=args.atlas_source,
        dk_atlas_ts_root=args.dk_atlas_ts_root,
    )
    n_subj = len(adapter.subject_ids)
    print(f"Biopoint adapter: {n_subj} subjects")

    # Infer min rest/task length so we don't request windows longer than the data
    min_rest_len = min(
        adapter.load_subject(sid).shape[0] for sid in adapter.subject_ids
    )
    min_task_len = min(
        adapter.load_task_subject(sid).shape[0] for sid in adapter.subject_ids
    )

    if args.prediction_length is None:
        args.prediction_length = min_task_len
        print(f"Inferred prediction_length={args.prediction_length}")
    args.prediction_length = min(args.prediction_length, min_task_len)

    lookback_length = min(args.lookback_length, min_rest_len)
    if lookback_length < 1:
        lookback_length = 1
    if args.prediction_length < 1:
        args.prediction_length = 1
    args.lookback_length = lookback_length

    print(f"Using lookback_length={args.lookback_length}, prediction_length={args.prediction_length} (data: min_rest={min_rest_len}, min_task={min_task_len})")

    train_ds = BiopointWindowDataset(
        adapter,
        lookback_length=args.lookback_length,
        prediction_length=args.prediction_length,
        stride=args.stride,
        normalize=args.normalize,
        split="train",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    val_ds = BiopointWindowDataset(
        adapter,
        lookback_length=args.lookback_length,
        prediction_length=args.prediction_length,
        stride=args.stride,
        normalize=args.normalize,
        split="val",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    test_ds = BiopointWindowDataset(
        adapter,
        lookback_length=args.lookback_length,
        prediction_length=args.prediction_length,
        stride=args.stride,
        normalize=args.normalize,
        split="test",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    if args.normalize and train_ds.rest_means is not None:
        val_ds.rest_means, val_ds.rest_stds = train_ds.rest_means, train_ds.rest_stds
        val_ds.task_means, val_ds.task_stds = train_ds.task_means, train_ds.task_stds
        test_ds.rest_means, test_ds.rest_stds = train_ds.rest_means, train_ds.rest_stds
        test_ds.task_means, test_ds.task_stds = train_ds.task_means, train_ds.task_stds

    if len(train_ds) == 0:
        raise ValueError(
            f"No train windows: lookback_length={args.lookback_length} and prediction_length={args.prediction_length} "
            f"may exceed your data (min_rest={min_rest_len}, min_task={min_task_len}). "
            "Try smaller --lookback_length and --prediction_length."
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    sample = next(iter(train_loader))
    V = int(sample["input"].shape[-1])
    print(f"Windows: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}  V={V}")
    return train_loader, val_loader, test_loader, V


def run_timegan(args, train_loader, val_loader, test_loader, V):
    from baselines.train_timegan import (
        TimeGAN,
        train_embedder_recovery,
        train_supervisor,
        train_joint,
        evaluate_subject_level_dedup_timegan,
    )
    from baselines.eval_viz import save_closest_subject_visualizations
    from baselines.train_timegan import aggregate_subject_timeline
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from tqdm import tqdm

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join(args.save_dir, "timegan")
    os.makedirs(save_dir, exist_ok=True)

    model = TimeGAN(
        input_dim=V,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_prediction_length=args.prediction_length,
    ).to(device)

    embedder_opt = Adam(
        list(model.embedder.parameters()) + list(model.recovery.parameters()),
        lr=args.lr, weight_decay=1e-5,
    )
    supervisor_opt = Adam(model.supervisor.parameters(), lr=args.lr, weight_decay=1e-5)
    g_opt = Adam(
        list(model.generator.parameters()) + list(model.recovery.parameters()),
        lr=args.lr, weight_decay=1e-5,
    )
    d_opt = Adam(model.discriminator.parameters(), lr=args.lr * 0.5, weight_decay=1e-5)
    g_sched = CosineAnnealingLR(g_opt, T_max=args.epochs)
    d_sched = CosineAnnealingLR(d_opt, T_max=args.epochs)

    best_path = os.path.join(save_dir, "best_timegan.pth")
    if args.eval_only:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_metrics, subj_pred, subj_tgt, subj_starts, subj_total_len, per_subj = evaluate_subject_level_dedup_timegan(
            model, test_loader, device, args.prediction_length, return_for_viz=True
        )
        _print_test_metrics("TimeGAN", test_metrics)
        save_closest_subject_visualizations(
            per_subj, subj_pred, subj_tgt, subj_starts, subj_total_len,
            aggregate_subject_timeline, out_dir=save_dir, model_name="TimeGAN", fs=0.72,
        )
        _write_test_results(save_dir, "TimeGAN", test_metrics)
        return

    for ep in range(1, args.embedder_epochs + 1):
        loss = train_embedder_recovery(model, train_loader, embedder_opt, device, args.prediction_length)
        if ep % 5 == 0:
            print(f"  [E/R] Epoch {ep}/{args.embedder_epochs}  loss={loss:.6f}")
    for ep in range(1, args.supervisor_epochs + 1):
        loss = train_supervisor(model, train_loader, supervisor_opt, device, args.prediction_length)
        if ep % 5 == 0:
            print(f"  [SUP] Epoch {ep}/{args.supervisor_epochs}  loss={loss:.6f}")

    best_val = float("inf")
    history = []
    for ep in range(1, args.epochs + 1):
        g_loss, d_loss = train_joint(
            model, train_loader, g_opt, d_opt, device, args.prediction_length,
            lambda_embed=args.lambda_embed, lambda_supervise=args.lambda_supervise,
            lambda_adv=args.lambda_adv, lambda_psd=args.freq_loss_weight,
        )
        val_metrics = evaluate_subject_level_dedup_timegan(model, val_loader, device, args.prediction_length)
        g_sched.step()
        d_sched.step()
        print(f"Epoch {ep}/{args.epochs}  g_loss={g_loss:.6f}  d_loss={d_loss:.6f}  val_mse={val_metrics['mse']:.6f}")
        history.append({"epoch": ep, "g_loss": g_loss, "d_loss": d_loss, "val_mse": val_metrics["mse"]})
        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            torch.save({
                "epoch": ep, "model_state_dict": model.state_dict(),
                "args": vars(args), "best_val_mse": best_val,
            }, best_path)
            print(f"  Saved best -> {best_path}")

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics, subj_pred, subj_tgt, subj_starts, subj_total_len, per_subj = evaluate_subject_level_dedup_timegan(
        model, test_loader, device, args.prediction_length, return_for_viz=True
    )
    _print_test_metrics("TimeGAN", test_metrics)
    save_closest_subject_visualizations(
        per_subj, subj_pred, subj_tgt, subj_starts, subj_total_len,
        aggregate_subject_timeline, out_dir=save_dir, model_name="TimeGAN", fs=0.72,
    )
    _write_test_results(save_dir, "TimeGAN", test_metrics)
    with open(os.path.join(save_dir, "history.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "g_loss", "d_loss", "val_mse"])
        w.writeheader()
        w.writerows(history)


def run_timevae(args, train_loader, val_loader, test_loader, V):
    from baselines.timevae_baseline import (
        TimeVAE,
        train_epoch,
        evaluate_subject_level_dedup,
    )
    from baselines.eval_viz import save_closest_subject_visualizations
    from baselines.timevae_baseline import aggregate_subject_timeline
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join(args.save_dir, "timevae")
    os.makedirs(save_dir, exist_ok=True)

    model = TimeVAE(
        input_dim=V,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        latent_dim=args.latent_dim,
        output_dim=V,
        max_prediction_length=max(args.prediction_length, 256),
        dropout=args.dropout,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_path = os.path.join(save_dir, "best_model.pth")

    if args.eval_only:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_metrics, subj_pred, subj_tgt, subj_starts, subj_total_len, per_subj = evaluate_subject_level_dedup(
            model, test_loader, device, prediction_length=args.prediction_length, beta=args.beta,
            compute_freq_diff=True, compute_fc_sim=True, return_for_viz=True,
        )
        _print_test_metrics("TimeVAE", test_metrics)
        save_closest_subject_visualizations(
            per_subj, subj_pred, subj_tgt, subj_starts, subj_total_len,
            aggregate_subject_timeline, out_dir=save_dir, model_name="TimeVAE", fs=0.72,
        )
        _write_test_results(save_dir, "TimeVAE", test_metrics)
        return

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        current_beta = args.beta * (epoch / args.beta_warmup_epochs) if epoch <= args.beta_warmup_epochs else args.beta
        current_tf = args.teacher_forcing_ratio * (1 - (epoch - 1) / (args.epochs // 2)) + 0.1 * ((epoch - 1) / (args.epochs // 2)) if epoch <= args.epochs // 2 else 0.1
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            prediction_length=args.prediction_length, beta=current_beta,
            teacher_forcing_ratio=current_tf, max_grad_norm=args.max_grad_norm,
            freq_loss_weight=args.freq_loss_weight,
        )
        val_metrics = evaluate_subject_level_dedup(
            model, val_loader, device, prediction_length=args.prediction_length, beta=current_beta,
            compute_freq_diff=True, compute_fc_sim=True,
        )
        scheduler.step()
        print(f"Epoch {epoch}/{args.epochs}  loss={train_metrics['loss']:.6f}  val_mse={val_metrics['mse']:.6f}")
        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val, "args": vars(args),
            }, best_path)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics, subj_pred, subj_tgt, subj_starts, subj_total_len, per_subj = evaluate_subject_level_dedup(
        model, test_loader, device, prediction_length=args.prediction_length, beta=args.beta,
        compute_freq_diff=True, compute_fc_sim=True, return_for_viz=True,
    )
    _print_test_metrics("TimeVAE", test_metrics)
    save_closest_subject_visualizations(
        per_subj, subj_pred, subj_tgt, subj_starts, subj_total_len,
        aggregate_subject_timeline, out_dir=save_dir, model_name="TimeVAE", fs=0.72,
    )
    _write_test_results(save_dir, "TimeVAE", test_metrics)


def run_diffusion_ts(args, train_loader, val_loader, test_loader, V):
    from baselines.diffusion_ts_baseline import (
        DiffusionTS,
        train_epoch,
        evaluate_subject_level_dedup,
        aggregate_subject_timeline,
    )
    from baselines.eval_viz import save_closest_subject_visualizations
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join(args.save_dir, "diffusion_ts")
    os.makedirs(save_dir, exist_ok=True)

    model = DiffusionTS(
        v_dim=V, num_timesteps=args.num_timesteps,
        rest_hidden=args.rest_hidden, ctx_dim=args.ctx_dim,
    ).to(device)
    opt = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    best_path = os.path.join(save_dir, "best_diffusion_ts.pth")

    if args.eval_only:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        test_metrics, subj_pred, subj_tgt, subj_starts, subj_total_len, per_subj = evaluate_subject_level_dedup(
            model, test_loader, device, args.prediction_length, args.sample_steps, return_for_viz=True
        )
        _print_test_metrics("Diffusion-TS", test_metrics)
        save_closest_subject_visualizations(
            per_subj, subj_pred, subj_tgt, subj_starts, subj_total_len,
            aggregate_subject_timeline, out_dir=save_dir, model_name="Diffusion-TS", fs=0.72,
        )
        _write_test_results(save_dir, "Diffusion-TS", test_metrics)
        return

    best_val = float("inf")
    for ep in range(1, args.epochs + 1):
        tr_loss = train_epoch(
            model, train_loader, opt, device, max_grad_norm=1.0,
            pred_len=args.prediction_length,
            freq_loss_weight=args.freq_loss_weight,
            freq_aux_steps=args.freq_aux_steps,
        )
        val_metrics = evaluate_subject_level_dedup(model, val_loader, device, args.prediction_length, args.sample_steps)
        sched.step()
        print(f"Epoch {ep}/{args.epochs}  train_loss={tr_loss:.6f}  val_mse={val_metrics['mse']:.6f}")
        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics, subj_pred, subj_tgt, subj_starts, subj_total_len, per_subj = evaluate_subject_level_dedup(
        model, test_loader, device, args.prediction_length, args.sample_steps, return_for_viz=True
    )
    _print_test_metrics("Diffusion-TS", test_metrics)
    save_closest_subject_visualizations(
        per_subj, subj_pred, subj_tgt, subj_starts, subj_total_len,
        aggregate_subject_timeline, out_dir=save_dir, model_name="Diffusion-TS", fs=0.72,
    )
    _write_test_results(save_dir, "Diffusion-TS", test_metrics)


def run_ddpm(args, train_loader, val_loader, test_loader, V):
    from baselines.ddpm_baseline import (
        DDPM,
        train_epoch,
        evaluate_subject_level_dedup,
        aggregate_subject_timeline,
    )
    from baselines.eval_viz import save_closest_subject_visualizations
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join(args.save_dir, "ddpm")
    os.makedirs(save_dir, exist_ok=True)

    model = DDPM(
        v_dim=V, num_timesteps=args.num_timesteps,
        rest_hidden=args.rest_hidden, ctx_dim=args.ctx_dim,
        beta_start=args.beta_start, beta_end=args.beta_end,
    ).to(device)
    opt = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    best_path = os.path.join(save_dir, "best_ddpm.pth")

    if args.eval_only:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        test_metrics, subj_pred, subj_tgt, subj_starts, subj_total_len, per_subj = evaluate_subject_level_dedup(
            model, test_loader, device, args.prediction_length, args.sample_steps, return_for_viz=True
        )
        _print_test_metrics("DDPM", test_metrics)
        save_closest_subject_visualizations(
            per_subj, subj_pred, subj_tgt, subj_starts, subj_total_len,
            aggregate_subject_timeline, out_dir=save_dir, model_name="DDPM", fs=0.72,
        )
        _write_test_results(save_dir, "DDPM", test_metrics)
        return

    best_val = float("inf")
    for ep in range(1, args.epochs + 1):
        tr_loss = train_epoch(
            model, train_loader, opt, device, args.max_grad_norm,
            pred_len=args.prediction_length,
            freq_loss_weight=args.freq_loss_weight,
            freq_aux_steps=args.freq_aux_steps,
        )
        val_metrics = evaluate_subject_level_dedup(model, val_loader, device, args.prediction_length, args.sample_steps)
        sched.step()
        print(f"Epoch {ep}/{args.epochs}  train_loss={tr_loss:.6f}  val_mse={val_metrics['mse']:.6f}")
        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics, subj_pred, subj_tgt, subj_starts, subj_total_len, per_subj = evaluate_subject_level_dedup(
        model, test_loader, device, args.prediction_length, args.sample_steps, return_for_viz=True
    )
    _print_test_metrics("DDPM", test_metrics)
    save_closest_subject_visualizations(
        per_subj, subj_pred, subj_tgt, subj_starts, subj_total_len,
        aggregate_subject_timeline, out_dir=save_dir, model_name="DDPM", fs=0.72,
    )
    _write_test_results(save_dir, "DDPM", test_metrics)


def run_lstm_gan(args, train_loader, val_loader, test_loader, V):
    from baselines.train_lstm_gan import (
        LSTMGAN,
        train_joint,
        evaluate_subject_level_dedup_lstm_gan,
    )
    from baselines.eval_viz import save_closest_subject_visualizations
    from baselines.train_timegan import aggregate_subject_timeline
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join(args.save_dir, "lstm_gan")
    os.makedirs(save_dir, exist_ok=True)

    model = LSTMGAN(
        input_dim=V,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_prediction_length=max(args.prediction_length, 256),
    ).to(device)

    g_opt = Adam(
        list(model.encoder.parameters()) + list(model.generator.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay if getattr(args, "weight_decay", None) is not None else 1e-5,
    )
    d_opt = Adam(
        model.discriminator.parameters(),
        lr=args.lr * 0.5,
        weight_decay=args.weight_decay if getattr(args, "weight_decay", None) is not None else 1e-5,
    )
    g_sched = CosineAnnealingLR(g_opt, T_max=args.epochs)
    d_sched = CosineAnnealingLR(d_opt, T_max=args.epochs)

    best_path = os.path.join(save_dir, "best_lstm_gan.pth")
    lambda_mse = getattr(args, "lambda_mse", 10.0)

    if args.eval_only:
        if not os.path.isfile(best_path):
            raise FileNotFoundError(f"eval_only: checkpoint not found at {best_path}")
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_metrics, subj_pred, subj_tgt, subj_starts, subj_total_len, per_subj = evaluate_subject_level_dedup_lstm_gan(
            model, test_loader, device, args.prediction_length, return_for_viz=True
        )
        _print_test_metrics("LSTM-GAN", test_metrics)
        save_closest_subject_visualizations(
            per_subj, subj_pred, subj_tgt, subj_starts, subj_total_len,
            aggregate_subject_timeline, out_dir=save_dir, model_name="LSTM-GAN", fs=0.72,
        )
        _write_test_results(save_dir, "LSTM-GAN", test_metrics)
        return

    best_val = float("inf")
    for ep in range(1, args.epochs + 1):
        g_loss, d_loss = train_joint(
            model, train_loader, g_opt, d_opt, device, args.prediction_length, lambda_mse=lambda_mse
        )
        val_metrics = evaluate_subject_level_dedup_lstm_gan(model, val_loader, device, args.prediction_length)
        g_sched.step()
        d_sched.step()
        print(f"Epoch {ep}/{args.epochs}  g_loss={g_loss:.6f}  d_loss={d_loss:.6f}  val_mse={val_metrics['mse']:.6f}")
        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "best_val_mse": best_val,
            }, best_path)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics, subj_pred, subj_tgt, subj_starts, subj_total_len, per_subj = evaluate_subject_level_dedup_lstm_gan(
        model, test_loader, device, args.prediction_length, return_for_viz=True
    )
    _print_test_metrics("LSTM-GAN", test_metrics)
    save_closest_subject_visualizations(
        per_subj, subj_pred, subj_tgt, subj_starts, subj_total_len,
        aggregate_subject_timeline, out_dir=save_dir, model_name="LSTM-GAN", fs=0.72,
    )
    _write_test_results(save_dir, "LSTM-GAN", test_metrics)


def _print_test_metrics(name, metrics):
    print("\n" + "=" * 60)
    print(f"{name} TEST (Biopoint, subject-level dedup)")
    print("=" * 60)
    print(f"MSE:   {metrics['mse']:.6f} ± {metrics['mse_std']:.6f}")
    print(f"MAE:   {metrics['mae']:.6f} ± {metrics['mae_std']:.6f}")
    print(f"PSD:   {metrics['freq_diff']:.6f} ± {metrics['freq_diff_std']:.6f}")
    print(f"FC sim: {metrics['fc_similarity']:.6f} ± {metrics['fc_similarity_std']:.6f}")
    print(f"Num subjects: {metrics['num_subjects']}")
    print("=" * 60)


def _write_test_results(save_dir, name, metrics):
    path = os.path.join(save_dir, "test_results.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{name} TEST (Biopoint, subject-level dedup)\n")
        f.write("=" * 60 + "\n")
        f.write(f"MSE (mean ± std): {metrics['mse']:.6f} ± {metrics['mse_std']:.6f}\n")
        f.write(f"MAE (mean ± std): {metrics['mae']:.6f} ± {metrics['mae_std']:.6f}\n")
        f.write(f"PSD (mean ± std): {metrics['freq_diff']:.6f} ± {metrics['freq_diff_std']:.6f}\n")
        f.write(f"FC sim (mean ± std): {metrics['fc_similarity']:.6f} ± {metrics['fc_similarity_std']:.6f}\n")
        f.write(f"Num subjects: {metrics['num_subjects']}\n")
        f.write("=" * 60 + "\n")
    print(f"Test results written to {path}")


def main():
    args = get_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {args.model}  device: {args.device}  save_dir: {args.save_dir}")

    # If evaluating, prefer atlas settings stored in the checkpoint args.
    if args.eval_only:
        import torch

        ckpt_name = {
            "timegan": "best_timegan.pth",
            "timevae": "best_model.pth",
            "diffusion_ts": "best_diffusion_ts.pth",
            "ddpm": "best_ddpm.pth",
            "lstm_gan": "best_lstm_gan.pth",
        }.get(args.model)
        ckpt_subdir = {
            "timegan": "timegan",
            "timevae": "timevae",
            "diffusion_ts": "diffusion_ts",
            "ddpm": "ddpm",
            "lstm_gan": "lstm_gan",
        }.get(args.model)

        if ckpt_name and ckpt_subdir:
            best_path = os.path.join(args.save_dir, ckpt_subdir, ckpt_name)
            if os.path.isfile(best_path):
                try:
                    ckpt = torch.load(best_path, map_location="cpu")
                    saved_args = ckpt.get("args") or {}
                    args.atlas_source = saved_args.get("atlas_source", args.atlas_source)
                    args.dk_atlas_ts_root = saved_args.get("dk_atlas_ts_root", args.dk_atlas_ts_root)
                    print(f"Loaded atlas settings from checkpoint: atlas_source={args.atlas_source!r}")
                except Exception:
                    # Fall back to CLI defaults.
                    pass

    train_loader, val_loader, test_loader, V = build_dataloaders(args)

    if args.model == "timegan":
        run_timegan(args, train_loader, val_loader, test_loader, V)
    elif args.model == "timevae":
        run_timevae(args, train_loader, val_loader, test_loader, V)
    elif args.model == "diffusion_ts":
        run_diffusion_ts(args, train_loader, val_loader, test_loader, V)
    elif args.model == "ddpm":
        run_ddpm(args, train_loader, val_loader, test_loader, V)
    elif args.model == "lstm_gan":
        run_lstm_gan(args, train_loader, val_loader, test_loader, V)
    else:
        raise ValueError(f"Unknown model: {args.model}")


if __name__ == "__main__":
    main()
