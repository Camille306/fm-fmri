# Visualization Guide

The training script automatically generates visualization plots for the best model during training.

## What Gets Generated

When the best model is found (lowest validation loss), the script creates:

### 1. Individual ROI Plots (268 files)
- **Location**: `{save_dir}/visualizations_epoch_{best_epoch}/ROI_001.png` through `ROI_268.png`
- **Content**: Each plot shows:
  - **Input Context** (blue line): The lookback sequence used for prediction
  - **Ground Truth** (green circle): The actual next time step value
  - **Prediction** (red X): The model's prediction
  - **Metrics**: MSE, MAE, and RMSE for that specific ROI
  - **Legend**: Clearly labeled with all three elements

### 2. Summary Statistics Plot
- **Location**: `{save_dir}/visualizations_epoch_{best_epoch}/summary_statistics.png`
- **Content**: Four subplots showing:
  - Prediction vs Ground Truth scatter plot (all ROIs)
  - MSE per ROI (bar chart)
  - MAE per ROI (bar chart)
  - RMSE per ROI (bar chart)
  - Overall statistics in the title

## Plot Details

### Individual ROI Plots
- **X-axis**: Time steps (input context + prediction point)
- **Y-axis**: Signal amplitude (denormalized if normalization was used)
- **Vertical dashed line**: Separates input context from prediction point
- **Multiple samples**: Shows up to `--num_vis_samples` different validation samples per ROI
- **Legend**: 
  - "Input Context" (blue)
  - "Ground Truth" (green circle)
  - "Prediction" (red X)

### Summary Plot
- Provides an overview of model performance across all 268 brain regions
- Helps identify which ROIs are predicted well vs poorly
- Shows overall model performance metrics

## Usage

Visualizations are enabled by default. To disable:
```bash
python train.py --no-visualize
```

To change the number of samples shown per ROI:
```bash
python train.py --num_vis_samples 20
```

## Output Structure

```
checkpoints/
├── best_model.pth
├── checkpoint_epoch_5.pth
├── checkpoint_epoch_10.pth
└── visualizations_epoch_15/
    ├── ROI_001.png
    ├── ROI_002.png
    ├── ...
    ├── ROI_268.png
    └── summary_statistics.png
```

## Interpreting the Plots

### Good Predictions
- Red X (prediction) close to green circle (ground truth)
- Low MSE/MAE/RMSE values in the title
- Predictions follow the trend of the input context

### Poor Predictions
- Large gap between prediction and ground truth
- High error metrics
- May indicate:
  - Need for more training
  - Difficult ROI to predict
  - Model architecture limitations

### Summary Statistics
- **Scatter plot**: Points should cluster along the diagonal (red dashed line)
- **Bar charts**: Lower bars indicate better performance for that ROI
- Look for patterns: Are certain ROIs consistently harder to predict?

## Notes

- Visualizations are generated only when a new best model is found
- Data is automatically denormalized for visualization if normalization was used during training
- Plots are saved at 150 DPI for high quality
- The visualization folder is named after the epoch where the best model was found




