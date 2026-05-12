# FM-fMRI: Flow Matching for Rest-to-Task fMRI Prediction

This repository contains the code accompanying our work on predicting
task-evoked fMRI sequences from resting-state fMRI using conditional flow
matching, together with a suite of baselines (TimeGAN, TimeVAE, DDPM,
Diffusion-TS, LSTM-GAN, and Time-XL).

The released code targets the **HCP** (Human Connectome Project) setup: rest
-> task prediction on the Shen-268 atlas across the seven HCP tasks (emotion,
gambling, language, motor, relational, social, working memory).

> **A note on the Biopoint experiments.** The accompanying paper also reports
> results on an in-house task-fMRI dataset (Biopoint) for graph-based
> classification and rest-conditioned generation. Because that dataset is held
> by the originating lab and cannot be redistributed, the corresponding code
> (`biopoint/`, `gat_biopoint/`, `gcn_gat_biopoint/`, `stagin_biopoint/`, and
> the Biopoint-specific re-evaluation scripts) is **not** included in this
> public release. The HCP code in this repository is self-contained and
> reproduces the HCP results in the paper.
>
> Subject identifiers, lab-internal paths, and HPC-specific submission
> scripts have also been removed. Only source code is included; raw and
> derived data must be obtained separately (see "Data" below).

---

## Repository layout

```
fm-fmri/             Main flow-matching model (the primary method)
baselines/           Re-organized baseline implementations for the HCP setup
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

The code targets **Python 3.10** with PyTorch >= 2.0. A representative
environment:

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision torchaudio   # pick the CUDA build for your system
pip install numpy scipy pandas tqdm matplotlib scikit-learn h5py einops
pip install nibabel              # if you use task_preprocess/
```

The `time-xl/` subproject has its own `requirements.txt`.

---

## Data

The code expects two directories under `./data/` (paths are configurable via
`--data_root` and `--task_root` on every script):

| Default path                          | Contents                              |
|---------------------------------------|---------------------------------------|
| `./data/hcp-resting-fc/<task>/<sub>/` | HCP resting-state Shen-268 ROI series |
| `./data/hcp-task-ts/<task>/<sub>/`    | HCP task-fMRI Shen-268 ROI series     |

We do **not** redistribute HCP. Obtain it through
[ConnectomeDB](https://db.humanconnectome.org/) under the HCP Data Use Terms.

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

Outputs (per-task metrics, FC / PSD figures for the closest subject) are
written to `--save_dir` (default `./results_<baseline>_<task>`).

### 3. Re-evaluation

`re_eval/` builds JSON configs that point at a set of trained baselines, then
recomputes metrics (FC similarity, PSD difference, discriminative score,
cFID) on a common evaluation grid:

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

Per-baseline scripts save two qualitative figures for the closest test
subject (lowest MSE) -- `closest_subject_<id>_fc.png` and
`closest_subject_<id>_psd.png` -- produced by helpers in `eval_viz.py`.

---

## Reproducing the paper (HCP results)

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
- If you find a bug or have trouble reproducing a result, please open a
  GitHub issue.

---

## Citation

If this code is useful in your research, please cite the accompanying paper.
(BibTeX entry will be added on acceptance.)

## License

Released under the MIT License -- see `LICENSE`. The vendored `time-xl/`
subproject retains its own license; see `time-xl/LICENSE` (Apache 2.0).
