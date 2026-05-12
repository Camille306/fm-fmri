# FM-fMRI: Flow Matching for Rest-to-Task fMRI Prediction

This repository contains the code accompanying our work on predicting task-evoked
fMRI sequences from resting-state fMRI using conditional flow matching, together
with a suite of baselines (TimeGAN, TimeVAE, DDPM, Diffusion-TS, LSTM-GAN,
STAGIN, GAT, GCN+GAT, and Time-XL).

The repo covers two datasets:
- **HCP** (Human Connectome Project) rest -> task prediction on Shen-268 ROIs.
- **Biopoint** (in-house task-fMRI dataset) for graph-based classification and
  rest-conditioned generation.

> Subject identifiers, lab-internal paths, and HPC-specific scripts have been
> removed from this public release. Only the source code is included; raw and
> derived data must be obtained separately (see "Data" below).

---

## Repository layout

```
fm-fmri/             Main flow-matching model (the primary method)
baselines/           Re-organized baseline implementations for the HCP setup
biopoint/            Biopoint-dataset experiments and dataloaders
gat_biopoint/        GAT classifier on Biopoint
gcn_gat_biopoint/    GCN + GAT variant on Biopoint
stagin_biopoint/     STAGIN baseline on Biopoint
timegan/             TimeGAN baseline (HCP)
timeVAE/             TimeVAE baseline (HCP)
time-xl/             Time-XL (xLSTM / Mamba / LSTM) baselines
fm-ts/               Flow-matching time-series baseline (HCP, no EVs)
re_eval/             Re-evaluation utilities (FC / PSD / discriminative scores)
task_preprocess/     HCP task-fMRI preprocessing helpers
model_submission/    Submission collection script
dataset.py           HCPRestingFCDataset (shared across baselines)
dataset_timeseries.py / model.py / train.py / inference.py
                     Original rest-to-task pipeline used by several baselines
```

Each subfolder has its own `README.md` (or comments at the top of the main
script) describing what it does and the command-line flags it accepts.

---

## Installation

The code targets **Python 3.10** with PyTorch >= 2.0. A representative environment:

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision torchaudio   # pick the CUDA build for your system
pip install numpy scipy pandas tqdm matplotlib scikit-learn h5py einops
pip install nibabel              # if you use task_preprocess/
pip install torch-geometric      # for GAT / GCN / STAGIN baselines
```

The `time-xl/` subproject has its own `requirements.txt`.

---

## Data

The code expects a few directories under `./data/` (all paths are configurable
via `--data_root`, `--task_root`, etc., on every script):

| Default path                          | Contents                                 |
|---------------------------------------|------------------------------------------|
| `./data/hcp-resting-fc/<task>/<sub>/` | HCP resting-state Shen-268 ROI series    |
| `./data/hcp-task-ts/<task>/<sub>/`    | HCP task-fMRI Shen-268 ROI series        |
| `./data/biopoint_data/`               | Biopoint task-fMRI per-subject series    |
| `./data/biopoint_dk_atlas/`           | Biopoint Desikan-Killiany ROI series     |
| `./data/biopoint_data.csv`            | Biopoint subject metadata + labels       |

We do **not** redistribute either dataset. Obtain HCP data through
[ConnectomeDB](https://db.humanconnectome.org/) under the HCP Data Use Terms;
Biopoint data is held by the originating lab and is not publicly released.

Once you have the data, point the scripts at it:

```bash
python fm-fmri/fm_fmri.py \
    --data_root /your/path/to/hcp-resting-fc \
    --task_root /your/path/to/hcp-task-ts \
    --task_name emotion
```

---

## Quick start

### 1. Flow matching (main method)

```bash
python fm-fmri/fm_fmri.py --task_name emotion --epochs 50 --batch_size 16
```

Key flags (see `fm-fmri/METHOD_DETAILS.md` for the full architecture):

- `--rest_encoder {transformer,lstm}` -- context encoder for the rest signal
- `--use_evs` -- condition on event tables (onset / duration / condition_id)
- `--use_hrf_kernel` -- learnable HRF basis convolution on the rest input
- `--ode_steps 50` -- Euler steps for inference
- `--freq_loss_weight 0.1 --fc_loss_weight 0.1` -- auxiliary loss weights

### 2. HCP baselines

All baselines share the same dataloader and evaluation protocol:

```bash
python baselines/train_timegan.py        --task_name emotion
python baselines/timevae_baseline.py     --task_name emotion
python baselines/ddpm_baseline.py        --task_name emotion
python baselines/diffusion_ts_baseline.py --task_name emotion
python baselines/train_lstm_gan.py       --task_name emotion
```

Outputs (per-task metrics, FC / PSD figures for the closest subject) are written
to `--save_dir` (default `./results_<baseline>_<task>`).

### 3. Biopoint experiments

```bash
# GAT classifier on Desikan-Killiany ROIs
python gat_biopoint/main.py --data_root ./data/biopoint_dk_atlas

# STAGIN baseline
python stagin_biopoint/main.py

# Flow matching on Biopoint
python biopoint/run_flow_matching_biopoint.py
```

### 4. Re-evaluation

`re_eval/` builds JSON configs that point at a set of trained baselines, then
recomputes metrics (FC similarity, PSD difference, discriminative score, cFID)
on a common evaluation grid:

```bash
python re_eval/build_baselines_config.py --out re_eval/my_config.json
python re_eval/run_all_baselines_metrics.py --config re_eval/my_config.json
```

---

## Evaluation protocol (shared across HCP baselines)

All HCP-side methods use the same sliding-window protocol:

1. **Windowing.** From each subject's rest sequence, take a window of
   `lookback_length` TRs; predict the following `prediction_length` TRs of
   task-fMRI. Each window carries a `subject_id` and `task_start_idx`.
2. **Subject-level aggregation.** Overlapping predicted windows are averaged
   per subject to form a single deduplicated timeline.
3. **Metrics.** MSE, MAE, FC similarity (Pearson on the upper triangle of the
   correlation matrix), and PSD difference (mean ROI power spectrum).

Per-baseline scripts save two qualitative figures for the closest test subject
(lowest MSE) -- `closest_subject_<id>_fc.png` and `closest_subject_<id>_psd.png`
-- produced by helpers in `eval_viz.py`.

---

## Reproducing the paper

A representative end-to-end run on a single HCP task:

```bash
# Train all baselines + the main flow-matching model
python baselines/train_timegan.py        --task_name emotion --save_dir runs/timegan_emotion
python baselines/timevae_baseline.py     --task_name emotion --save_dir runs/timevae_emotion
python baselines/ddpm_baseline.py        --task_name emotion --save_dir runs/ddpm_emotion
python baselines/diffusion_ts_baseline.py --task_name emotion --save_dir runs/diffts_emotion
python baselines/train_lstm_gan.py       --task_name emotion --save_dir runs/lstmgan_emotion
python fm-fmri/fm_fmri.py                --task_name emotion --save_dir runs/fm_emotion

# Build a re-eval config and recompute metrics on a common grid
python re_eval/build_baselines_config.py --runs_root runs --out re_eval/emotion.json
python re_eval/run_all_baselines_metrics.py --config re_eval/emotion.json
```

Substitute `emotion` with any HCP task (`gambling`, `language`, `motor`,
`relational`, `social`, `wm`).

---

## Notes

- Defaults for `--data_root` etc. are placeholders (`./data/...`) -- set them
  to the actual location of your data, or override on the command line.
- The codebase originated on an HPC cluster; cluster-specific submission
  scripts (SLURM, rsync helpers) have been omitted from this release.
- If you find a bug or have trouble reproducing a result, please open a GitHub
  issue.

---

## Citation

If this code is useful in your research, please cite the accompanying paper.
(BibTeX entry will be added on acceptance.)

## License

A repository-wide `LICENSE` file has not been added yet. The vendored
`time-xl/` subproject retains its own license; see `time-xl/LICENSE` (Apache
2.0). Please add a top-level `LICENSE` before publishing the repo (MIT,
Apache 2.0, and BSD-3 are all common choices for research code).
