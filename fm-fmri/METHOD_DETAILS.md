# FM-fMRI: Method Details for Reviewers

## 1. Training setup

| Setting | Value | Notes |
|--------|--------|------|
| **Optimizer** | Adam | — |
| **Learning rate** | 1e-3 | Default; configurable via `--lr` |
| **Weight decay** | 1e-5 | L2 regularization |
| **Batch size** | 16 | Per-GPU; configurable via `--batch_size` |
| **Epochs** | 50 | Configurable via `--epochs` |
| **LR schedule** | Cosine annealing | `CosineAnnealingLR(optimizer, T_max=epochs)`; no warmup |
| **Gradient clipping** | Yes | `clip_grad_norm_(parameters(), max_grad_norm)` with **max_grad_norm = 1.0** (default) |
| **EMA** | No | No exponential moving average of parameters |

**Loss:** Flow-matching MSE (primary) plus optional auxiliary losses with configurable weights: frequency (PSD) loss (`freq_loss_weight`, default 0.1), FC (correlation matrix) loss (`fc_loss_weight`, default 0.1), and coherence loss (`coh_loss_weight`, default 0.0). FC loss can be weighted by connection strength (`fc_weight_by_strength`, default True; `fc_strength_power`, default 2.0). An auxiliary loss can optionally predict x₁ with fewer ODE steps (`aux_ode_steps`, default 10).

**Inference:** ODE integration from t=0→1 with Euler steps; default **ode_steps = 50** (configurable via `--ode_steps`).

---

## 2. Architecture specifics

### Rest encoder (context from resting-state)

- **Type:** Patch-based Transformer (default) or 2-layer LSTM; selected via `rest_encoder` (`"transformer"` | `"lstm"`).
- **Transformer variant:**
  - **Patch length:** 16 TRs (`rest_patch_len`). Rest sequence (B, L, V) is split into non-overlapping patches of length 16; each patch is flattened (patch_len × V) and projected to `d_model`.
  - **Depth:** 2 encoder layers (`rest_num_layers`).
  - **Heads:** 4 (`rest_nhead`).
  - **Model dimension:** 256 (`rest_hidden`); same as `d_model` for the transformer.
  - **FFN dimension:** 512 (`rest_dim_feedforward`). Activation: GELU; `batch_first=True`; no pre-norm (`norm_first=False`).
  - **Output:** A [CLS] token is prepended; positional embeddings are added; output context is the **CLS token embedding** projected to **ctx_dim = 256**.
- **LSTM variant:** 2 layers, hidden size 256, final hidden state projected to ctx_dim 256.

### Optional: HRF on rest input

- If **use_hrf_kernel** is True: before the rest encoder, a **learnable 1D convolution over time** is applied to (B, L, V). The kernel is a linear combination of a fixed **HRF basis** (gamma + temporal derivative + dispersion derivative; 3 basis functions, length 20 TRs by default). Weights are learned (shared across ROIs by default, or per-ROI with `hrf_per_roi`). This is **not** applied to the task/EV side by default; it only affects the rest input to the encoder.

### Velocity network (v_θ(t, x_t | rest [, EVs]))

- **Inputs (concatenated per timepoint):**  
  `x_t` (B, T, V), **rest context** (B, ctx_dim) broadcast to (B, T, ctx_dim), **time embedding** (B, t_dim) broadcast to (B, T, t_dim), and optionally **event context** from cross-attention (B, T, d_ev).
- **Time embedding:** Scalar t ∈ [0,1] → 2-layer MLP (Linear(1, t_dim) → SiLU → Linear(t_dim, t_dim) → SiLU); **t_dim = 128**.
- **Structure:** One hidden layer:  
  `Linear(in_dim, 512) → SiLU → Dropout(0.1) → Linear(512, 512) → SiLU → Dropout(0.1) → Linear(512, V)`.  
  **in_dim** = V + ctx_dim + t_dim (+ d_ev if EVs used). So the velocity network is an **MLP** (no extra transformer layers in vnet); **hidden size 512**, dropout 0.1.
- **Cross-attention (when EVs are used):** For each of the T timepoints, **Q** = Linear(x_t) → (B, T, d_ev); **K, V** = event tokens (B, N_events, d_ev). Standard scaled dot-product attention over events (masked for padding); output (B, T, d_ev) is concatenated with x_t, rest_ctx, and t_emb before the MLP. So **cross-attention is applied once per forward pass**, with queries from the current state x_t and keys/values from the event representations (see EV handling below).

### Prior head (rest-conditioned x₀)

- **Low-rank structure:** From rest context (B, ctx_dim), a **PriorHead** outputs:  
  - **mean** (B, V), **std** (B, V): per-ROI location and scale (std clamped to [0.1, 2.0]).  
  - **U** (B, V, **K**): low-rank factor; **K = 8** by default (`prior_K`).  
- **Sampling:** z ~ N(0,1) of shape (B, T, K); correlated component = z @ U^T → (B, T, V); **x₀ = mean + std * ε + (z @ U^T)** with ε being per-ROI Gaussian (optionally 1/f noise). So **K is the low-rank dimension** for the rest-conditioned prior over the trajectory.

### Dimensions summary

| Symbol / name | Default | Description |
|---------------|---------|-------------|
| V | data | Number of ROIs (e.g. 400) |
| ctx_dim | 256 | Rest context dimension |
| rest_hidden | 256 | Transformer d_model / LSTM hidden |
| t_dim | 128 | Time embedding dimension |
| d_ev | 64 | Event token dim and cross-attention dim |
| prior_K | 8 | Low-rank factor size for prior |
| vnet hidden | 512 | Velocity MLP hidden size (fixed in code) |

---

## 3. EV handling: mapping EVs to per-timepoint conditioning

### EV data format

- **Input:** EV table of shape **(N_events, 4)** per subject: **[onset (TR), duration (TR), amplitude, condition_id]**. Condition 0 is reserved for padding and is masked out in attention.
- **Usage:** EVs are **not** inherently per-timepoint; they are event-level. Per-timepoint conditioning is achieved either by **cross-attention** (event tokens as K/V, trajectory timepoints as Q) or by **HRF-convolved event timecourses** that modulate event K/V per timepoint.

### Two modes (mutually exclusive in the velocity network)

**A. Event tokens only (no HRF timecourse)**  
- **EVEncoder:** First 3 columns (onset, duration, amplitude) → MLP → d_ev; 4th column (condition_id) → embedding → d_ev; **sum** → event token (B, N_events, d_ev).  
- **Per-timepoint conditioning:** For each of the T timepoints, the velocity net has **Q = Linear(x_t)** and **K = V = event_tokens**. So each timepoint **attends over the same set of event tokens**; the only “temporal” signal is that x_t changes along the trajectory. There is **no HRF convolution** in this mode.

**B. HRF timecourse-conditioned event K/V (optional)**  
- **Flag:** `use_ev_hrf_timecourse` (off by default).  
- **Event-level:** From EV table, **onset** and **duration** define a boxcar (or smooth sigmoid boxcar if `ev_hrf_smooth_boxcar`) in **window-relative TRs** (if `task_start_idx` is provided, onset is shifted by task start). Optional per-event **delay** and **width** are predicted from event tokens (`ev_hrf_use_delay_width`, default True) and applied to onset/duration.  
- **HRF convolution:** The boxcar is convolved with a **per-event HRF kernel**. The kernel is a **learned linear combination of the same gamma-derived basis** (num_basis=3, kernel_len=20 TRs by default). Weights are computed from the event token (so each event gets its own HRF-shaped timecourse). Result is a **timecourse (B, N_events, T)** normalized to max 1 per event.  
- **Per-timepoint conditioning:** **K** remains time-invariant: K = projK(event_tokens). **V** is made time-dependent: **V_tc(b,t,n,:) = event_token(b,n,:) * timecourse(b,n,t)**. The velocity net still uses **Q = Linear(x_t)**; attention is over events, but the **values** are scaled by the HRF timecourse at each timepoint t. So “EVs mapped to per-timepoint” here means: **HRF-convolved event timecourses** modulate the value vectors that each timepoint receives from each event; no HRF is applied to the BOLD rest input unless `use_hrf_kernel` is set.

### Summary

- **EVs → per-timepoint:** Either (1) **cross-attention only**: same event K/V for all timepoints (no HRF), or (2) **EV HRF timecourse**: same event keys, **values** scaled by a per-event, HRF-convolved timecourse so that conditioning varies by timepoint.  
- **HRF usage:**  
  - **Rest:** Optional learnable HRF-basis convolution on rest **before** the rest encoder (`use_hrf_kernel`).  
  - **EVs:** Optional **only** when `use_ev_hrf_timecourse` is True: HRF is used to build **event-level timecourses** that weight the event **values** in cross-attention; no HRF is applied to raw BOLD for the EV path.

---

## Reference: key defaults (fm_fmri.py)

```text
# Training
--batch_size 16 --epochs 50 --lr 1e-3 --weight_decay 1e-5 --max_grad_norm 1.0
# No EMA; scheduler: CosineAnnealingLR(T_max=epochs)

# Architecture
--rest_encoder transformer --rest_hidden 256 --ctx_dim 256 --rest_patch_len 16
--rest_num_layers 2 --rest_nhead 4 --rest_dim_feedforward 512
--prior_K 8 --t_dim 128
# VelocityNet: hidden=512, dropout=0.1

# EVs (when --use_evs)
--num_conditions 32 --d_ev 64
# Optional: --use_ev_hrf_timecourse, ev_hrf_kernel_len 20, ev_hrf_num_basis 3

# Optional rest HRF
# --use_hrf_kernel --hrf_kernel_len 20 --hrf_num_basis 3 [--hrf_per_roi]
```
