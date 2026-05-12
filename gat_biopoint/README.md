# GAT Biopoint

Autism classification from Biopoint resting-state fMRI using a **Graph Attention Network (GAT)** in the style of [STNAGNN-fMRI](https://github.com/Jiyao96/STNAGNN-fMRI). Supports **fMRI synthetic data augmentation** (train with real + synthetic samples, test on real only), matching the setup used in `stagin_biopoint`.

## Setup

- Python 3.8+
- PyTorch, torch-geometric, scikit-learn, pandas, numpy
- Same Biopoint data layout as `stagin_biopoint`: default `sourcedir` `./data_pi_lab/user/project/biopoint_data`, ROI files `output/{subject_id}/rest/{subject_id}_shen268_ts.npy`, CSV with `subject_id` and `group` (pat/control).

## Usage

**Real data only (single train/test split, default 80/20):**
```bash
python main.py --train --test
```

**With synthetic fMRI augmentation:**
```bash
python main.py --train --test --use_synthetic --synthetic_dir /path/to/synthetic_biopoint
```

Synthetic data: same format as STAGIN (`synthetic_manifest.csv` with `subject_id`, `label`, `path`; `*_syn.npy`). Only synthetics from **training** subjects are used (test-subject synthetics excluded to avoid leakage).

## Options

- `-k`, `--k_fold`: folds (default 1 = single stratified train/test split)
- `--train_ratio`: train fraction when k_fold=1 (default 0.8)
- `--window_size`, `--window_stride`, `--window_num`: dynamic FC windows (default 50, 3, 12)
- `--hidden_dim`, `--dropout`, `--num_epochs`, `--lr`: model and training
- `--dynamic_length`: optional fixed time length
- `--ts_filename_suffix`: default `_shen268_ts.npy` (Biopoint); use `_aal_ts.npy` for AAL

## Output

- Trained models: `targetdir/model/{fold}/model.pth`
- Test metrics (accuracy, F1, AUC) per fold and mean, written to `targetdir/test_results.txt`
