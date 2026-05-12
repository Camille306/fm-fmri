# LSTM Baseline Model Logic Explanation

## Overview
The `lstm_baseline.py` implements a **Rest-to-Task fMRI Prediction** model. It predicts task fMRI data from resting-state fMRI data using an LSTM architecture.

## Input and Output

### Input
- **Shape**: `(batch_size, seq_len, input_dim)`
  - `batch_size`: Number of samples in the batch
  - `seq_len`: Fixed sequence length = `lookback_length` (default: 512 time steps)
  - `input_dim`: Number of brain regions/ROIs (default: 268)
- **Content**: Resting-state fMRI time series data
  - Each sample is a sliding window from a subject's resting-state data
  - The window has a fixed length of 512 time points
  - Each time point has 268 features (one per brain region)

### Output
- **If `prediction_length == 1`**:
  - Shape: `(batch_size, output_dim)`
  - Single time step prediction for all 268 brain regions
- **If `prediction_length > 1`**:
  - Shape: `(batch_size, prediction_length, output_dim)`
  - Sequence prediction: `prediction_length` time steps for all 268 brain regions

## Model Architecture

```python
LSTMBaseline(
    input_dim=268,      # Number of brain regions
    hidden_dim=128,     # LSTM hidden dimension
    num_layers=2,       # Number of LSTM layers
    output_dim=268,     # Output dimension (same as input)
    dropout=0.1         # Dropout rate
)
```

**Components:**
1. **LSTM layers**: Process the input sequence
   - Input: `(batch_size, lookback_length=512, input_dim=268)`
   - Output: `(batch_size, 512, hidden_dim=128)`
2. **Fully connected layer**: Maps hidden state to output
   - Input: `(batch_size, hidden_dim=128)`
   - Output: `(batch_size, output_dim=268)`

## Forward Pass Logic

### Single Step Prediction (`prediction_length == 1`)
```python
# 1. LSTM processes the full input sequence
lstm_out, (h_n, c_n) = self.lstm(x)  
# lstm_out shape: (batch_size, 512, 128)

# 2. Take the last hidden state (from the last time step)
last_hidden = lstm_out[:, -1, :]  
# last_hidden shape: (batch_size, 128)

# 3. Project to output dimension
output = self.fc(last_hidden)  
# output shape: (batch_size, 268)
```

### Multi-Step Prediction (`prediction_length > 1`)
```python
# 1. LSTM processes the full input sequence
lstm_out, (h_n, c_n) = self.lstm(x)  
# lstm_out shape: (batch_size, 512, 128)

# 2. Take the last hidden state
last_hidden = lstm_out[:, -1, :]  
# last_hidden shape: (batch_size, 128)

# 3. Repeat the same hidden state for each prediction time step
hidden_repeated = last_hidden.unsqueeze(1).repeat(1, prediction_length, 1)  
# hidden_repeated shape: (batch_size, prediction_length, 128)

# 4. Project to output dimension
output = self.fc(hidden_repeated)  
# output shape: (batch_size, prediction_length, 268)
```

**Note**: For multi-step prediction, the model uses the **same hidden state** for all prediction steps. This is a simple baseline approach - all predicted time steps are based on the same context.

## How It Handles Various Input Lengths

### Fixed Input Length per Sample
- Each individual sample has a **fixed input length** of `lookback_length` (512 time steps)
- The LSTM processes exactly 512 time steps per sample

### Variable Input via Sliding Windows
The model handles "various lengths" through **sliding windows**:

1. **Sliding Window Creation**:
   ```python
   # For each subject's resting-state data (e.g., 1200 time points)
   # Create multiple windows with stride
   for rest_start_idx in range(0, max_windows + 1, stride):
       # Window 1: indices [0:512]
       # Window 2: indices [100:612]  (if stride=100)
       # Window 3: indices [200:712]
       # ...
   ```

2. **Stride Parameter**:
   - `stride=1`: Every possible window (many samples, slower)
   - `stride=100`: Every 100th window (fewer samples, faster)
   - Default: `stride=100` for efficiency

3. **Different Starting Positions**:
   - Each window starts at a different position in the subject's data
   - This creates diversity in the training data
   - All windows have the same length (512), but capture different temporal contexts

### Example
For a subject with 1200 time points of resting-state data:
- With `lookback_length=512` and `stride=100`:
  - Window 1: time points [0:512]
  - Window 2: time points [100:612]
  - Window 3: time points [200:712]
  - Window 4: time points [300:812]
  - ...
  - Window 7: time points [600:1112]
  - Window 8: time points [700:1200] (last possible window)

Each window is a separate training sample with the same input length (512).

## Data Flow

### Training/Evaluation Flow

1. **Dataset Creation** (`FMRIWindowDataset`):
   - Creates sliding windows from resting-state data
   - Each window: `(lookback_length=512, num_variables=268)`
   - Target: corresponding task data `(prediction_length, 268)`

2. **DataLoader**:
   - Batches multiple windows together
   - Input batch: `(batch_size, 512, 268)`
   - Target batch: `(batch_size, 268)` or `(batch_size, prediction_length, 268)`

3. **Model Forward Pass**:
   - Input: `(batch_size, 512, 268)` → LSTM → `(batch_size, 512, 128)`
   - Last hidden: `(batch_size, 128)` → FC → `(batch_size, 268)`

4. **Loss Computation**:
   - Compare prediction `(batch_size, 268)` with target `(batch_size, 268)`
   - Use MSE loss (or frequency loss)

## Key Points

1. **Fixed Input Length**: Each sample always has exactly 512 time steps
2. **Variable Context**: Different windows capture different temporal contexts from the same subject
3. **Sliding Window Strategy**: Creates multiple training samples from one subject's data
4. **Simple Multi-Step Prediction**: For `prediction_length > 1`, repeats the same hidden state (not autoregressive)

## Limitations

1. **Non-Autoregressive Multi-Step**: When `prediction_length > 1`, all predictions use the same hidden state, so they're identical (just repeated). This is a simple baseline.

2. **Fixed Context Window**: The model only sees the last 512 time steps of resting-state data, not the full history.

3. **No Temporal Evolution**: For multi-step predictions, the model doesn't evolve the hidden state over prediction steps.

## Potential Improvements

1. **Autoregressive Generation**: Use previous predictions as input for next time step
2. **Decoder LSTM**: Add a decoder LSTM for better multi-step prediction
3. **Attention Mechanism**: Allow the model to attend to different parts of the input sequence
4. **Variable Length Inputs**: Use padding and masking to handle truly variable-length sequences
