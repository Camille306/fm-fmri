# STAGIN on Biopoint (Autism Classification)

This folder trains [STAGIN](https://github.com/egyptdj/stagin) (Spatio-Temporal Attention Graph Isomorphism Network, NeurIPS 2021) on the Biopoint dataset for **autism vs control** classification using ROI time series (default: **Shen268**, same as biopoint fm-fmri).

## Data layout

- **Data root** (`-ds` / `--sourcedir`): directory containing Biopoint outputs (same as `biopoint/run_flow_matching_biopoint.py`: default `./data/biopoint_data`).
- **ROI time series**: for each subject, rest scans are read from  
  `{sourcedir}/output/{subject_id}/rest/{subject_id}_shen268_ts.npy`  
  with shape `(T, num_roi)` (Shen268 → 268 ROIs). Default suffix: `_shen268_ts.npy`.
- **Labels**: CSV with columns `subject_id` and `group` (e.g. `pat` / `con`).  
  Default CSV: `.//fMRI_autism/v3_causality_lstm/causal_lstm_2025/biopoint_data.csv`. Override with `--csv_path`.

For AAL or another atlas, set `--ts_filename_suffix _aal_ts.npy` (and ensure files exist under `output/<id>/rest/<id>_aal_ts.npy`).

## Setup

From repo root (or with `stagin_biopoint` on `PYTHONPATH`):

```bash
cd stagin_biopoint
pip install torch numpy pandas scikit-learn einops tqdm tensorboard
```

## Run

**Train (and optionally validate each epoch):**

```bash
python main.py --train --validate \
  -ds /path/to/biopoint_data \
  -dt /path/to/results \
  -n stagin_biopoint_aal \
  -k 5 \
  -b 8
```

**Test (after training):**

```bash
python main.py --test \
  -ds /path/to/biopoint_data \
  -dt /path/to/results \
  -n stagin_biopoint_aal
```

**Train + test in one go (default if no flag is given):**

```bash
python main.py -ds /path/to/biopoint_data -dt /path/to/results
```

## Options

| Option | Default | Description |
|--------|--------|-------------|
| `-ds` / `--sourcedir` | `./biopoint_data` | Biopoint data root (contains `output/<id>/rest/`) |
| `-dt` / `--targetdir` | `./result` | Results directory (models, metrics, attention) |
| `-n` / `--exp_name` | `stagin_biopoint` | Experiment name (subfolder under `targetdir`) |
| `--csv_path` | `{sourcedir}/biopoint_data.csv` | Subject list CSV (`subject_id`, `group`) |
| `--ts_filename_suffix` | `_aal_ts.npy` | ROI file suffix → `{subject_id}<suffix>` |
| `--dynamic_length` | None | Fixed window length (default: use full time series) |
| `-k` | 5 | Cross-validation folds |
| `-b` | 8 | Minibatch size |
| `--window_size` | 50 | Sliding window for dynamic FC |
| `--window_stride` | 3 | Stride for sliding window |
| `--num_epochs` | 40 | Training epochs per fold |
| `--num_heads` | 1 | Transformer heads |
| `--num_layers` | 4 | STAGIN layers |
| `--hidden_dim` | 128 | Hidden dimension |
| `--sparsity` | 30 | Graph sparsity (percentile) |

Outputs: trained models per fold, `metric.csv`, TensorBoard logs under `summary/`, and attention/latent under `attention/`.

## Real vs Real+Synthetic comparison (fm-fmri augmentation)

To test whether adding **fm-fmri–generated synthetic** task time series improves STAGIN autism classification:

1. **Train fm-fmri** on Biopoint (rest→task) and save a checkpoint, e.g. `results_flow_matching_biopoint/best_fmts.pth`.
2. **Generate synthetic** task time series from that checkpoint (one per subject):
   ```bash
   python generate_synthetic_biopoint.py \
     --data_root /path/to/biopoint_data \
     --fm_checkpoint /path/to/results_flow_matching_biopoint/best_fmts.pth \
     --synthetic_dir ./synthetic_biopoint \
     --ts_filename_suffix _shen268_ts.npy
   ```
   This writes `synthetic_dir/{subject_id}_syn.npy` and `synthetic_manifest.csv`.
3. **Run the comparison** (trains STAGIN twice: real-only, then real+synthetic; evaluates and compares):
   ```bash
   python run_real_vs_synthetic_comparison.py \
     --data_root /path/to/biopoint_data \
     --result_dir ./comparison_results \
     --fm_checkpoint /path/to/results_flow_matching_biopoint/best_fmts.pth \
     --generate
   ```
   Use `--generate` to create synthetic if missing. Outputs: `result_dir/real_only/`, `result_dir/real_plus_synthetic/`, and `result_dir/comparison.csv` with metric deltas.

Alternatively, train with synthetic manually:
- **Real-only:** `python main.py --train --test -ds ... -dt result_dir/real_only`
- **Real+synthetic:** `python main.py --train --test -ds ... -dt result_dir/real_plus_synthetic --use_synthetic --synthetic_dir ./synthetic_biopoint`

## Reference

- STAGIN: [Learning Dynamic Graph Representation of Brain Connectome with Spatio-Temporal Attention](https://arxiv.org/abs/2105.13495) (NeurIPS 2021), [code](https://github.com/egyptdj/stagin).
