# FM-fMRI: Flow Matching for Rest-to-Task fMRI Prediction

This repository contains the code accompanying our work on predicting
task-evoked fMRI sequences from resting-state fMRI using conditional flow
matching.

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
> **A note on baselines.** Re-implementations of comparison baselines used in
> the paper (TimeGAN, TimeVAE, DDPM, Diffusion-TS, LSTM-GAN, Time-XL) are not
> included in this release. Each of those methods has its own canonical
> implementation by the original authors, and we refer readers to those
> repositories.
>
> Subject identifiers, lab-internal paths, and HPC-specific submission
> scripts have also been removed. Only source code is included; raw and
> derived data must be obtained separately (see "Data" below).

---

## Repository layout

```
fm-fmri/             Main flow-matching model (the primary method)
task_preprocess/     HCP task-fMRI preprocessing helpers
model_submission/    Submission collection script
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

```bash
python fm-fmri/fm_fmri.py --task_name emotion --epochs 50 --batch_size 16
```

Key flags (see `fm-fmri/METHOD_DETAILS.md` for the full architecture):

- `--rest_encoder {transformer,lstm}` -- context encoder for the rest signal
- `--use_evs` -- condition on event tables (onset / duration / condition_id)
- `--ode_steps 50` -- Euler steps for inference
- `--freq_loss_weight 0.1 --fc_loss_weight 0.1` -- auxiliary loss weights

---

## Evaluation protocol

The model uses a sliding-window protocol:

1. **Windowing.** From each subject's rest sequence, take a window of
   `lookback_length` TRs; predict the following `prediction_length` TRs of
   task-fMRI. Each window carries a `subject_id` and `task_start_idx`.
2. **Subject-level aggregation.** Overlapping predicted windows are averaged
   per subject to form a single deduplicated timeline.
3. **Metrics.** MSE, MAE, FC similarity (Pearson on the upper triangle of the
   correlation matrix), and PSD difference (mean ROI power spectrum).

The evaluation script saves two qualitative figures for the closest test
subject (lowest MSE) -- `closest_subject_<id>_fc.png` and
`closest_subject_<id>_psd.png` -- produced by helpers in `eval_viz.py`.

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

Released under the MIT License -- see `LICENSE`.
