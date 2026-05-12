# Re-eval: Unified metrics for rest-to-task fMRI generation

This folder contains re-evaluation scripts that use the same conditional pipeline: load the trained model and test set, run inference to get **paired** (real, generated) task windows (same rest conditioning per pair).

---

## 1. Discriminative score (LSTM classifier)

Evaluates how distinguishable generated task time series are from real using an LSTM binary classifier.

### Usage

```bash
python re_eval/run_discriminative_score.py --load_dir /path/to/best/model --task_name emotion
```

`--data_root` and `--task_root` default to HCP (or `DATA_ROOT` / `TASK_ROOT` env vars). Override if your data lives elsewhere.

Optional: `--use_evs` if the model was trained with event files; `--save_dir` to write the score and classifier checkpoint.

### Output

- Printed: validation accuracy and discriminative score.
- If `--save_dir` is set: `discriminative_score.txt` and `lstm_classifier.pth`.

### Data leakage check (e.g. if you get discriminative score 0.5)

- **Real** and **generated** come only from the **test set** (test_loader has `split="test"`; train/val subjects are never used).
- The LSTM is trained on a **random subset** of the mixed (real, generated) samples and validated on the **rest**; the split is **disjoint** (no sample in both train and val), with a **fixed seed** for reproducibility.
- If you see score 0.5 (perfect separation), run with **`--debug_scale`** to print mean/std of real vs generated; a large scale mismatch would make them trivially separable and is worth fixing in the model or data pipeline.

---

## 2. cFID-FC (conditional Fréchet distance on FC)

Measures the Fréchet distance between the **distributions** of functional connectivity (FC) features of real vs generated task signals, on the **same paired samples** (conditional evaluation).

- **Feature**: \(f(x) = \mathrm{vec}(\mathrm{FC}(x))\) where \(\mathrm{FC}(x)\) is the V×V correlation matrix of time series \(x\) (T×V), and \(\mathrm{vec}\) is the upper triangle (excluding diagonal), so \(d = V(V-1)/2\).
- **Formula**:  
  \(\mathrm{cFID\_FC} = \|\mu_r - \mu_g\|^2 + \mathrm{Tr}(C_r + C_g - 2(C_r C_g)^{1/2})\)  
  with \(\mu_r, C_r\) and \(\mu_g, C_g\) the mean and covariance of \(f(\text{real})\) and \(f(\text{generated})\) over the paired set. Covariances are regularized with \(C \leftarrow C + \epsilon I\) (\(\epsilon=10^{-6}\)) for numerical stability.
- **Interpretation**: Lower cFID-FC = better (network topology and variability/diversity realism).

### Usage

```bash
python re_eval/run_cfid_fc.py --load_dir /path/to/best/model --task_name emotion
```

`--data_root` and `--task_root` default to HCP (or `DATA_ROOT` / `TASK_ROOT`). Optional: `--use_evs`, `--eps`, `--save_dir` (writes `cfid_fc.txt`).  
By default cFID-FC uses `--max_fc_dim 500` (random projection) for speed with large V (e.g. 166 ROIs). Use `--max_fc_dim 0` for exact cFID on full FC dimension (slower and heavier).

---

## 3. Re-evaluate all best baselines (unified metrics)

Runs **MAE**, **PSD**, **FC similarity**, **FC top 5% precision**, **cFID-FC**, and **discriminative score** for every checkpoint listed in a config file. All metrics are computed on the same paired (real, generated) test windows.

- **MAE**: Mean absolute error between real and generated time series (window-level average).
- **PSD**: Mean absolute power-spectrum difference (window-level average, 0.01–0.05 Hz band).
- **FC similarity**: Pearson correlation between upper-triangle FC of real vs generated (window-level average).
- **FC top 5% precision**: Of predicted top 5% edges, fraction that are in ground-truth top 5% (window-level average).
- **cFID-FC**: Conditional Fréchet distance on FC features (see above).
- **Discriminative score**: LSTM classifier |validation accuracy − 0.5| (0 = indistinguishable, 0.5 = perfectly distinguishable).

### Config format

JSON array of objects with:

- `name`: Display name for the run.
- `load_dir`: Directory containing `best_fmts.pth`.
- `task_name` (optional): Override global `--task_name` for this entry.

**Building the config on the cluster:** If checkpoints live under  
`.//{experiment_name}/runs/*/best_fmts.pth`, run:

```bash
python re_eval/build_baselines_config.py --base_dir ./ --out re_eval/baselines_config.json
```

This scans experiments `nhead8_ablation_emotion`, `nhead8_ev_all_tasks`, `nhead8_ablation_other_tasks` by default. Use `--experiments ...` to override. Optionally `--no_task_name` to omit per-entry task (use CLI `--task_name` only).

Example (manual): `re_eval/baselines_config.example.json`

### Usage

```bash
python re_eval/run_all_baselines_metrics.py \
    --config re_eval/baselines_config.json \
    --out_csv re_eval/results_baselines.tsv
```

`task_name` comes from each config entry (the JSON built by `build_baselines_config.py` includes it). You only need `--task_name` as a fallback if some entries omit it. `--data_root` and `--task_root` default to HCP. Optional: `--use_evs`, `--classifier_epochs`, `--ode_steps`, etc.

### Where re-evaluation results are saved

- **Unified metrics (all baselines):** Only if you pass `--out_csv <path>`. The script writes a single **tab-separated** file at that path (e.g. `re_eval/results_baselines.tsv`), with columns: `name`, `mae`, `psd`, `fc_similarity`, `fc_precision_at_5`, `cfid_fc`, `discriminative_score`. One row per checkpoint. If you omit `--out_csv`, results are only printed to stdout.
- **Single-checkpoint scripts:** `run_discriminative_score.py` and `run_cfid_fc.py` save only when you pass `--save_dir <dir>`: the first writes `discriminative_score.txt` and `lstm_classifier.pth` there, the second writes `cfid_fc.txt` there.

---

## 4. Evaluate baselines only (TimeGAN, TimeVAE, DDPM, Diffusion-TS)

The unified script in §3 only runs **FM-TS** checkpoints (`best_fmts.pth`). To re-evaluate **only** the baseline models (timegan, timevae, ddpm, diffusion_ts) across all tasks:

**Where baselines look for checkpoints:** `baselines/slurm/run_baseline.sh` and `baselines/slurm/collect_results.py` use **`<repo_root>/results/<model>/<task>/`** (e.g. `results/timegan/emotion/`). The config built below uses the same layout by default. Use `--results_base` when building the config or when running `run_baseline_only_metrics.py` if your results live elsewhere.

1. **Build a baseline-only config** (from repo root or from `re_eval/re_eval/`):

   ```bash
   cd re_eval/re_eval && python build_combined_config.py --baseline_only
   ```
   This writes `re_eval/re_eval/fm_baseline_only.json` (28 entries: 4 models × 7 tasks). Custom results root: `python build_combined_config.py --baseline_only --results_base /path/to/results`.

2. **Run baseline eval and collect metrics** (from repo root):

   ```bash
   python re_eval/run_baseline_only_metrics.py \
       --config re_eval/re_eval/fm_baseline_only.json \
       --out_csv re_eval/results_baseline_only.tsv
   ```

   For each entry this runs the corresponding baseline script with `--eval_only` and `--save_dir <load_dir>` (each `load_dir` must contain the trained checkpoint), then parses `test_results.txt` and appends one row to the TSV. Columns match the unified metrics TSV (`name`, `mae`, `psd`, `fc_similarity`, `fc_precision_at_5`; `cfid_fc` and `discriminative_score` are left empty for baselines). If checkpoints are under a different results root: add `--results_base /path/to/repo/results`. Use `--data_root` and `--task_root` if your data is not at the default HCP paths.

   **Why cFID-FC and discriminative score are not computed for baselines:** The unified pipeline in §3 (`run_all_baselines_metrics.py`) is the only place that computes cFID-FC and discriminative score. It does so by loading the model, running inference on the test set to get paired (real, generated) windows, then calling `cfid_fc(X_real, X_gen)` and the LSTM classifier. That path currently supports **only FM-TS** (it uses `load_fmts_and_dataset`). Baseline entries are skipped there. The “baseline only” script above does not load baselines in Python; it runs each baseline’s own `--eval_only`, which only writes MSE, MAE, PSD, FC similarity (and top-k) to `test_results.txt`. So cFID-FC and discriminative score are never computed for timegan/timevae/ddpm/diffusion_ts unless we add baseline model loading and inference to the unified pipeline (same test loader → same metrics).

   If `test_results.txt` already exists for every entry (e.g. you ran each baseline’s eval before), you can skip re-running and only collect:  
   `python re_eval/run_baseline_only_metrics.py --config re_eval/re_eval/fm_baseline_only.json --out_csv re_eval/results_baseline_only.tsv --skip_run`

---

## 5. Biopoint runs (FM-TS and baselines)

Biopoint uses a different dataset and paths. Re-eval scripts above (§3–§4) use **HCP** data. For Biopoint:

**Where Biopoint results live (same as `biopoint/slurm`):** FM-TS: `results_flow_matching_biopoint/` or `results_fm_biopoint_sweep_freq_fc/runs/<name>/` or `results_flow_matching_biopoint_ablations/runs/<name>/`. Baselines: `results_biopoint/<model>/` (e.g. `results_biopoint/timegan/`).

**Build Biopoint config** (from `re_eval/re_eval/`): `python build_biopoint_config.py` → `biopoint.json`. Options: `--no_fm`, `--no_baselines`, `--fm_base`, `--sweep_base`, `--ablations_base`, `--results_biopoint`.

**Run Biopoint baseline-only metrics** (from repo root):  
`python re_eval/run_biopoint_baseline_only_metrics.py --config re_eval/re_eval/biopoint.json --out_csv re_eval/results_biopoint_baseline_only.tsv`  
Uses `--data_root`, `--csv_path`, `--results_biopoint`; `--skip_run` to only collect.

**Re-eval Biopoint FM-fMRI** (Flow Matching entries in `biopoint.json`):  
`python re_eval/run_biopoint_fm_only_metrics.py --config re_eval/re_eval/biopoint.json --out_csv re_eval/results_biopoint_fm_only.tsv`  
For each entry with `model_type: "fmts"` (e.g. `biopoint_fm/single`, `biopoint_fm/sweep/<name>`), runs `biopoint/run_flow_matching_biopoint.py --eval_only --save_dir <load_dir>`, then parses `test_results.txt` and writes one TSV. Uses `--data_root`, `--csv_path`; `--skip_run` to only collect from existing `test_results.txt`.

**Full unified re-eval on Biopoint** (cFID-FC, discriminative, etc.) is not in `run_all_baselines_metrics.py` (HCP only). Use `biopoint/slurm/collect_baselines_biopoint.py` and `biopoint/slurm/collect_ablations_flow_matching.py` for Biopoint.

---

## 6. Group-level FC comparison (real vs FM-fMRI vs baseline)

**`run_group_fc_comparison.py`** picks the **best** FM-fMRI and baseline checkpoints for a given task by reading **`test_results.txt`** in each config entry’s `load_dir` and selecting the entry with the **highest FC top-5% precision** (tiebreak: FC similarity). It then runs both models on the same test set and plots **group-level average Functional Connectivity** matrices: real (group avg), FM-fMRI generated (group avg), and baseline generated (group avg).

Configs: **`fm_fmri.json`** and **`fm_baseline_only.json`** (each entry’s `load_dir` must contain a trained checkpoint; for “best” selection, `load_dir/test_results.txt` should exist with FC similarity and k=5% precision lines).

**Usage** (from repo root):

```bash
python re_eval/run_group_fc_comparison.py \
  --task_name emotion \
  --fm_fmri_config re_eval/re_eval/fm_fmri.json \
  --baseline_config re_eval/re_eval/fm_baseline_only.json \
  --save_dir re_eval/group_fc_plots
```

- **`--task_name`** (required): task to compare (e.g. `emotion`, `WM`).
- **`--baseline_model_type`**: restrict baseline to one type (e.g. `timevae`). Only **timevae** is supported for baseline inference; others can be added later.
- **`--save_npy`**: also save the three group FC matrices as `.npy` files.

If no `test_results.txt` is found for any task-matching entry, the first entry is used and a message is printed.

Output: `group_fc_comparison_<task_name>.png` in `--save_dir` (three panels: Real | FM-fMRI | Baseline).
