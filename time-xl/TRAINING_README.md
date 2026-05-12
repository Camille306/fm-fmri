# Timer-XL Training Script for HCP Resting-State fMRI Data

This directory contains a training script (`train.py`) for fine-tuning Timer-XL on your HCP resting-state functional connectivity dataset.

## Overview

The training script implements:
- **Multivariate Next Token Prediction**: Adapts Timer-XL for multivariate time series (fMRI data with 268 brain regions)
- **Sliding Window Dataset**: Creates training windows from fMRI timeseries
- **Data Normalization**: Per-variable normalization for stable training
- **Subject-based Splitting**: Splits data by subjects (train/val/test)

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install torch>=2.0.0 transformers==4.40.1 numpy>=1.21.0
```

## Usage

### Basic Training

```bash
python train.py --data_root ./data/hcp-resting-fc --lookback_length 512 --batch_size 8 --epochs 50 --learning_rate 1e-4 --save_dir .
```

### Full Options

```bash
python train.py \
    --data_root ./data/hcp-resting-fc \
    --lookback_length 512 \
    --prediction_length 1 \
    --stride 1 \
    --normalize \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --model_name thuml/timer-base-84m \
    --from_pretrained \
    --batch_size 8 \
    --epochs 50 \
    --learning_rate 1e-4 \
    --weight_decay 1e-5 \
    --clip_grad_norm 1.0 \
    --save_dir ./checkpoints \
    --save_every 5 \
    --device cuda \
    --num_workers 4
```

## Arguments

### Data Arguments
- `--data_root`: Root directory containing subject folders (default: `./data/hcp-resting-fc`)
- `--lookback_length`: Number of time steps to use as input context (default: 512)
- `--prediction_length`: Number of time steps to predict (default: 1, for next token prediction)
- `--stride`: Stride for sliding window (default: 1). Increase to reduce memory usage
- `--normalize`: Normalize data to zero mean and unit variance (default: True)
- `--train_ratio`: Proportion of subjects for training (default: 0.7)
- `--val_ratio`: Proportion of subjects for validation (default: 0.15)
- `--max_samples_per_subject`: Maximum windows per subject (default: None = all). Set to limit memory usage
- `--norm_sample_size`: Number of samples to use for computing normalization stats (default: 10000)

### Model Arguments
- `--model_name`: HuggingFace model name (default: `thuml/timer-base-84m`)
- `--from_pretrained`: Load pretrained model from HuggingFace (default: True)

### Training Arguments
- `--batch_size`: Batch size for training (default: 8)
- `--epochs`: Number of training epochs (default: 50)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay for optimizer (default: 1e-5)
- `--clip_grad_norm`: Gradient clipping norm, 0 to disable (default: 1.0)
- `--save_dir`: Directory to save checkpoints (default: `./checkpoints`)
- `--save_every`: Save checkpoint every N epochs (default: 5)
- `--device`: Device to use (`cuda`/`cpu`), auto-detect if not specified
- `--num_workers`: Number of data loader workers (default: 4)

## Dataset Structure

The script expects your dataset to follow this structure:
```
data_root/
  subject_001/
    timeseries/
      REST1RL_Shen268_ts.npy
  subject_002/
    timeseries/
      REST1RL_Shen268_ts.npy
  ...
```

Each `.npy` file should contain a timeseries array of shape `(time_points, num_variables)` where:
- `time_points`: Number of time steps
- `num_variables`: Number of brain regions (268 for Shen268 atlas)

## Model Adaptation Notes

**Important**: The pre-trained Timer-XL model (`thuml/timer-base-84m`) is designed for univariate time series. For multivariate fMRI data:

1. **Input Reshaping**: The script flattens multivariate input `(batch, time, variables)` to `(batch, time*variables)` for compatibility.

2. **Output Projection**: The model's output may need adjustment to match the number of variables. The script attempts to handle this automatically.

3. **Fine-tuning**: Starting from a pre-trained model helps, but you may need to:
   - Adjust the model architecture if the default doesn't support your data dimensions
   - Use a custom model implementation from [OpenLTM](https://github.com/thuml/OpenLTM) if needed

## Output

The training script saves:
- `best_model.pth`: Best model checkpoint (lowest validation loss)
- `checkpoint_epoch_N.pth`: Periodic checkpoints every `--save_every` epochs

Each checkpoint contains:
- Model state dictionary
- Optimizer state dictionary
- Scheduler state dictionary
- Training/validation loss
- Training arguments

## Loading a Trained Model

```python
import torch
from transformers import AutoModelForCausalLM

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pth')
model = AutoModelForCausalLM.from_pretrained('thuml/timer-base-84m', trust_remote_code=True)
model.load_state_dict(checkpoint['model_state_dict'])

# Use for inference
model.eval()
# ... inference code ...
```

## Troubleshooting

### Out of Memory / "Killed" Error

If you get a "Killed" error during data loading, the dataset is too large for available memory. Try:

1. **Increase stride** to reduce number of windows:
   ```bash
   python train.py --stride 10  # Creates 10x fewer windows
   ```

2. **Limit windows per subject**:
   ```bash
   python train.py --max_samples_per_subject 100  # Max 100 windows per subject
   ```

3. **Reduce batch size**:
   ```bash
   python train.py --batch_size 4
   ```

4. **Reduce lookback length**:
   ```bash
   python train.py --lookback_length 256
   ```

5. **Use fewer subjects** (modify dataset.py or filter subject list)

### Memory-Efficient Implementation

The dataset now uses **lazy loading** - it only stores window metadata (subject_id, start_idx) instead of loading all data upfront. Data is loaded on-demand during training, which significantly reduces memory usage.

- Normalization stats are computed from a sample (default: 10,000 windows) instead of all windows
- Each window is loaded only when needed by the DataLoader
- This allows training on datasets with thousands of subjects without running out of memory

### Model Forward Pass Errors
- The model interface might differ from expected. Check the Timer-XL model implementation.
- Consider using the full OpenLTM repository for more control over the model architecture.

### Slow Training
- Increase `--num_workers` if you have multiple CPU cores
- Use `--stride > 1` to reduce number of training windows
- Consider using mixed precision training (not implemented in current script)

## References

- Timer-XL Paper: [arXiv:2410.04803](https://arxiv.org/abs/2410.04803)
- OpenLTM Repository: [https://github.com/thuml/OpenLTM](https://github.com/thuml/OpenLTM)
- HuggingFace Model: [thuml/timer-base-84m](https://huggingface.co/thuml/timer-base-84m)

