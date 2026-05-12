# TimeGAN Baseline for Rest-to-Task fMRI Prediction

This folder contains a TimeGAN (Time-series Generative Adversarial Networks) baseline implementation for predicting task fMRI sequences from resting-state fMRI sequences.

## Overview

TimeGAN is a generative adversarial network specifically designed for time series data. This implementation adapts TimeGAN for the rest-to-task prediction task with the following components:

- **Embedder**: Maps real data sequences to latent space
- **Recovery**: Maps latent sequences back to data space
- **Generator**: Generates synthetic task sequences from rest sequences
- **Discriminator**: Distinguishes real from synthetic task sequences
- **Supervisor**: Predicts next step in latent space (for temporal consistency)

## Files

- `timegan_model.py`: TimeGAN model architecture implementation
- `train_timegan.py`: Training script for TimeGAN baseline
- `README.md`: This file

## Installation

The TimeGAN implementation uses the same dependencies as the other baselines. Make sure you have:

```bash
pip install torch numpy scipy tqdm
```

## Usage

### Basic Training

```bash
python train_timegan.py \
    --task_root /path/to/hcp-task-ts \
    --task_name emotion \
    --data_root /path/to/hcp-resting-fc
```

### Full Training Options

```bash
python train_timegan.py \
    --data_root ./data/hcp-resting-fc \
    --task_root ./data/hcp-task-ts \
    --task_name emotion \
    --lookback_length 512 \
    --prediction_length 176 \
    --stride 100 \
    --normalize \
    --hidden_dim 64 \
    --num_layers 2 \
    --dropout 0.1 \
    --batch_size 32 \
    --epochs 50 \
    --embedder_epochs 10 \
    --supervisor_epochs 10 \
    --learning_rate 1e-3 \
    --lambda_embed 1.0 \
    --lambda_supervise 1.0 \
    --lambda_adv 1.0 \
    --save_dir ./checkpoints_timegan
```

## Training Phases

The TimeGAN training consists of three phases:

1. **Phase 1: Embedder/Recovery Pretraining** (`--embedder_epochs`)
   - Trains the embedder and recovery networks to learn good latent representations
   - Uses reconstruction loss (MSE between original and recovered data)

2. **Phase 2: Supervisor Pretraining** (`--supervisor_epochs`)
   - Trains the supervisor network to predict next steps in latent space
   - Ensures temporal consistency in the latent space

3. **Phase 3: Joint Training** (`--epochs`)
   - Jointly trains generator and discriminator
   - Combines adversarial loss, embedding loss, and supervised loss
   - Loss weights controlled by `--lambda_adv`, `--lambda_embed`, `--lambda_supervise`

## Arguments

### Data Arguments
- `--data_root`: Root directory containing subject folders for resting-state data
- `--task_root`: Root directory for task data
- `--task_name`: Name of the task (e.g., "emotion")
- `--lookback_length`: Number of time steps to use as input context (default: 512)
- `--prediction_length`: Number of time steps to predict (None = infer from task data)
- `--stride`: Stride for sliding window (default: 100)
- `--normalize`: Normalize data to zero mean and unit variance (default: True)
- `--train_ratio`: Proportion of subjects for training (default: 0.7)
- `--val_ratio`: Proportion of subjects for validation (default: 0.15)

### Model Arguments
- `--hidden_dim`: Hidden dimension for LSTM layers (default: 64)
- `--num_layers`: Number of LSTM layers (default: 2)
- `--dropout`: Dropout rate (default: 0.1)
- `--max_prediction_length`: Maximum supported prediction length (default: 256)

### Training Arguments
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of epochs for joint training (default: 50)
- `--embedder_epochs`: Number of epochs for embedder/recovery pretraining (default: 10)
- `--supervisor_epochs`: Number of epochs for supervisor pretraining (default: 10)
- `--learning_rate`: Learning rate (default: 1e-3)
- `--weight_decay`: Weight decay for optimizer (default: 1e-5)
- `--lambda_embed`: Weight for embedding loss in joint training (default: 1.0)
- `--lambda_supervise`: Weight for supervised loss in joint training (default: 1.0)
- `--lambda_adv`: Weight for adversarial loss in joint training (default: 1.0)
- `--save_dir`: Directory to save checkpoints (default: ./checkpoints_timegan)

## Output

The training script saves:
- `best_timegan_model.pth`: Best model checkpoint (based on validation MSE)
- `best_timegan_model_history.csv`: Training history with losses and metrics

## Evaluation Metrics

The model is evaluated using:
- **MSE** (Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **Frequency Difference**: Difference in power spectral density
- **Functional Connectivity Similarity**: Correlation between FC matrices

## Notes

- TimeGAN training can be more unstable than standard supervised learning due to the adversarial training
- You may need to tune the loss weights (`lambda_*`) for your specific dataset
- The embedder/recovery pretraining phase is important for stable training
- Larger `hidden_dim` may improve quality but increases memory usage

## References

- Original TimeGAN paper: "Time-series Generative Adversarial Networks" (Yoon et al., NeurIPS 2019)
- Adapted for rest-to-task fMRI prediction task
