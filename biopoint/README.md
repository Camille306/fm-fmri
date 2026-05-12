# Biopoint Rest-to-Task Baselines

Train and evaluate the same baseline models as in `baselines/` (TimeGAN, TimeVAE, Diffusion-TS, DDPM) on **Biopoint** matched rest–task data (Shen 268 ROI time series).

## Data

- **`biopoint_dataset.py`**: `fMRI_Biopoint_Dataset` returns `(rest, task, label)` per subject; `BiopointDatasetAdapter` exposes `subject_ids`, `load_subject()`, `load_task_subject()` for the baseline pipeline.
- **`biopoint_window_dataset.py`**: Sliding windows over rest/task with the same batch format as baselines: `input`, `target`, `subject_id`, `task_start_idx`.
- **`dataset.py`**: Shim so that baseline code `from dataset import HCPRestingFCDataset` can resolve when running from this folder (used when importing baseline modules).

## Running locally (from repo root)

```bash
# TimeGAN
python biopint/run_baselines_biopoint.py --model timegan --data_root /path/to/biopoint_data --save_dir ./results_biopoint

# TimeVAE, Diffusion-TS, DDPM
python biopint/run_baselines_biopoint.py --model timevae --data_root /path/to/biopoint_data --save_dir ./results_biopoint
python biopint/run_baselines_biopoint.py --model diffusion_ts --data_root /path/to/biopoint_data --save_dir ./results_biopoint
python biopint/run_baselines_biopoint.py --model ddpm --data_root /path/to/biopoint_data --save_dir ./results_biopoint
```

Results are written under `--save_dir/<model>/` (e.g. `results_biopoint/timegan/`): best checkpoint, `test_results.txt`, and FC/PSD plots for the closest subject.

**Eval only** (no training):

```bash
python biopint/run_baselines_biopoint.py --model timegan --save_dir ./results_biopoint --eval_only
```

## SLURM (cluster)

From `biopint/slurm/`:

```bash
# Single job
sbatch --job-name=timegan_biopoint --export=MODEL=timegan run_baseline.slurm

# All four models
./submit_all_biopoint.sh
```

Optional env vars: `DATA_ROOT`, `RESULTS_BASE` (defaults: `./data_pi_lab/user/project/biopoint_data`, `$REPO_ROOT/results_biopoint`).

## Options

- `--lookback_length`, `--prediction_length`, `--stride`: windowing (defaults 200, inferred from data, 10).
- `--train_ratio`, `--val_ratio`: 0.7 / 0.15 by default.
- `--batch_size`, `--epochs`, `--lr`, and model-specific flags (e.g. `--hidden_dim`, `--embedder_epochs` for TimeGAN) as in the main baselines.

Evaluation matches the baselines: subject-level aggregation of overlapping predicted windows, then MSE, MAE, PSD difference, and FC similarity (mean ± std across subjects).
