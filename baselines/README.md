# Baselines for Rest-to-Task fMRI Prediction

This folder contains baseline models for predicting task fMRI sequences from resting-state fMRI, with subject-level deduplicated evaluation (MSE, MAE, FC similarity, PSD difference). All scripts live directly in the **baselines/** folder.

## Scripts (directly in baselines/)

| Script | Description |
|--------|-------------|
| **train_timegan.py** | TimeGAN training. Uses **timegan_model.py** (Embedder, Recovery, Generator, Discriminator, Supervisor). Phased pretrain + joint training. |
| **timegan_model.py** | TimeGAN model definition (required by train_timegan.py). |
| **test_timegan.py** | Quick test: instantiate TimeGAN and run forward pass. |
| **timevae_baseline.py** | TimeVAE: Variational autoencoder (encoder → latent → decoder) for rest-to-task prediction. |
| **diffusion_ts_baseline.py** | Diffusion-TS: Conditional diffusion for time series. Rest-conditioned noise prediction; sample with reverse diffusion (DDIM-style). |
| **ddpm_baseline.py** | DDPM: Denoising Diffusion Probabilistic Models (Ho et al.). Rest-conditioned epsilon prediction; sample with full reverse process. |

## Running from project root

Dataset is loaded from the project root `dataset.py` (HCPRestingFCDataset). Run from project root:

```bash
# TimeGAN
python baselines/train_timegan.py --data_root /path/to/rest --task_root /path/to/task --task_name emotion

# TimeVAE
python baselines/timevae_baseline.py --data_root /path/to/rest --task_root /path/to/task --task_name emotion

# Diffusion-TS
python baselines/diffusion_ts_baseline.py --data_root /path/to/rest --task_root /path/to/task --task_name emotion

# DDPM
python baselines/ddpm_baseline.py --data_root /path/to/rest --task_root /path/to/task --task_name emotion
```

Or from inside **baselines/**:

```bash
cd baselines
python train_timegan.py --data_root /path/to/rest --task_root /path/to/task --task_name emotion
python timevae_baseline.py ...
python diffusion_ts_baseline.py ...
python ddpm_baseline.py ...
```

## Shared evaluation

All baselines use the same evaluation protocol:

- Sliding windows: rest (lookback_length) → task (prediction_length), with subject_id and task_start_idx.
- Subject-level aggregation: overlapping predicted windows are averaged per subject to form a single timeline.
- Metrics: MSE, MAE, frequency difference (PSD), FC similarity (Pearson on upper-triangle of correlation matrices).

## PSD (power spectrum) auxiliary loss

All baselines support an optional **PSD loss** during training to better match the power spectrum of the predicted signal to the target:

- **DDPM / Diffusion-TS**: `--freq_loss_weight` (default 0), `--freq_aux_steps` (default 10). A deterministic reverse pass is run each step to get a prediction, then MSE between PSD(pred) and PSD(target) is added to the loss.
- **TimeGAN**: `--freq_loss_weight` (default 0). PSD loss is added to the generator loss in joint training using the recovered sequence.
- **TimeVAE**: `--freq_loss_weight` (default 0). PSD loss is added to the VAE loss each step.

Shared implementation: `baselines/aux_losses.py` (`frequency_loss_torch`).

## Evaluation visualizations

When a baseline is evaluated on the test set, it saves two figures for the **closest subject** (lowest MSE) into `--save_dir`:

- **`closest_subject_{id}_fc.png`** — Ground truth vs predicted functional connectome (correlation matrices), with FC similarity in the title.
- **`closest_subject_{id}_psd.png`** — Average ROI power spectrum (GT vs predicted) and PSD difference.

These use the shared **eval_viz.py** helpers (`plot_fc_gt_vs_pred`, `plot_psd_spectrum_difference`, `save_closest_subject_visualizations`).

## Dependencies

- PyTorch, numpy, scipy, tqdm (same as the rest of the repo).
