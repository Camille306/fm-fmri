# Dual Loss Training and Visualization Script

## Overview

The `train_and_visualize_dual_loss.py` script trains the LSTM baseline model with two different loss functions and creates comprehensive visualizations comparing their performance.

## What It Does

1. **Trains with MSE Loss**
   - Saves checkpoint as: `best_model_mse.pth`
   - Uses standard Mean Squared Error loss

2. **Trains with Frequency Loss**
   - Saves checkpoint as: `best_model_freq.pth`
   - Uses frequency domain loss (FFT-based)

3. **Evaluates Both Models**
   - Finds the subject with highest FC similarity for each model
   - Generates predictions on test set

4. **Creates Visualizations** (for each model):
   - **Task FC Comparison**: Side-by-side comparison of predicted vs real functional connectivity matrices
   - **Frequency Spectrum**: Power spectral density comparison and difference plot

## Usage

### Basic Usage

```bash
python Timer-XL-main/train_and_visualize_dual_loss.py \
    --data_root /path/to/hcp-resting-fc \
    --task_root /path/to/hcp-task-ts \
    --task_name emotion \
    --epochs 50 \
    --save_dir ./checkpoints_dual_loss
```

### Skip Training (Visualization Only)

If you already have trained models and just want to create visualizations:

```bash
python Timer-XL-main/train_and_visualize_dual_loss.py \
    --data_root /path/to/hcp-resting-fc \
    --task_root /path/to/hcp-task-ts \
    --task_name emotion \
    --save_dir ./checkpoints_dual_loss \
    --skip_training
```

This will look for:
- `./checkpoints_dual_loss/best_model_mse.pth`
- `./checkpoints_dual_loss/best_model_freq.pth`

## Output Files

### Checkpoints
- `best_model_mse.pth`: Best model trained with MSE loss
- `best_model_freq.pth`: Best model trained with frequency loss

### Visualizations (in `save_dir/visualizations/`)
- `fc_comparison_mse_{subject_id}.png`: FC matrices for MSE model
- `fc_comparison_freq_{subject_id}.png`: FC matrices for Frequency model
- `frequency_spectrum_mse_{subject_id}.png`: Power spectrum for MSE model
- `frequency_spectrum_freq_{subject_id}.png`: Power spectrum for Frequency model

## Visualization Details

### Task FC Comparison
- Shows predicted and real functional connectivity matrices side-by-side
- Uses correlation matrices computed from task time series
- Displays FC similarity score
- Color-coded correlation values (RdBu colormap)

### Frequency Spectrum
- **Top plot**: Overlaid power spectral densities (predicted vs real)
- **Bottom plot**: Difference between predicted and real PSD
- Computed using Welch's method
- Shows frequency content preservation

## Key Parameters

- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 1e-3)
- `--lookback_length`: Input sequence length (default: 512)
- `--prediction_length`: Output sequence length (None = full task sequence)
- `--stride`: Sliding window stride (default: 100)

## Notes

- The script trains two separate models from scratch
- Both models use the same architecture and hyperparameters
- Only the loss function differs between the two
- The "closest subject" is determined by highest FC similarity score
- All visualizations are saved as high-resolution PNG files (300 DPI)

## Example Output Structure

```
checkpoints_dual_loss/
├── best_model_mse.pth
├── best_model_freq.pth
└── visualizations/
    ├── fc_comparison_mse_100307.png
    ├── fc_comparison_freq_100307.png
    ├── frequency_spectrum_mse_100307.png
    └── frequency_spectrum_freq_100307.png
```
