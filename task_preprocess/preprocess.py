#!/usr/bin/env python3
"""
Preprocess task fMRI files from a flat image directory.
Input:  {images_base}/{task}/images/{run}/*_tfMRI_{TASK}_{run}.nii.gz
Output: {output_base}/{task}/{sub_id}/{sub_id}_{atlas_name}_ts.npy
"""
from pathlib import Path
import argparse
import re
import sys
import warnings

import numpy as np
from nilearn.maskers import NiftiLabelsMasker
from tqdm import tqdm

# Suppress the deprecated import warning if anything still triggers it
warnings.filterwarnings("ignore", message=".*nilearn.input_data.*", category=FutureWarning)

# Valid HCP task names (used in dirs and in filenames as uppercase, except WM stays WM)
TASK_NAMES = ("emotion", "gambling", "language", "motor", "relational", "social", "WM")

# Base paths (will be formatted with task name)
IMAGES_BASE = Path("/home/user/palmer_scratch")
OUTPUT_BASE = Path("./data/palmer_scratch")
atlas_dir = Path("./data/data")

# Default task and run (overridable via CLI)
DEFAULT_TASK = "gambling"
DEFAULT_RUN = "RL"

# Atlases
atlases = {
    "AAL3": atlas_dir / "AAL3v1.nii.gz",
}


def get_paths_and_pattern(task: str, run: str):
    """Build images_dir, output_dir, and filename pattern for the given task and run."""
    images_dir = IMAGES_BASE / task / "images" / run
    output_dir = OUTPUT_BASE / task
    task_upper = task if task == "WM" else task.upper()
    pattern = re.compile(rf"^(\d+)_tfMRI_{re.escape(task_upper)}_(?:LR|RL)\.nii\.gz$")
    return images_dir, output_dir, pattern


def extract_timeseries_with_masker(masker, nifti_path: Path):
    """
    Extract ROI time series using an already-created masker (transform only).
    Returns:
      ts_valid: (T, N_valid)
      valid_cols: boolean mask of valid ROI columns (non-NaN)
    """
    ts = masker.transform(str(nifti_path))  # (T, N_all)
    valid_cols = ~np.isnan(ts).any(axis=0)
    ts_valid = ts[:, valid_cols]
    if ts_valid.shape[1] == 0:
        raise ValueError(f"No valid ROI time series from {nifti_path}")
    return ts_valid, valid_cols


def main(task: str = DEFAULT_TASK, run: str = DEFAULT_RUN):
    if task not in TASK_NAMES:
        raise ValueError(f"task must be one of {TASK_NAMES}, got {task!r}")
    images_dir, output_dir, task_fmri_pattern = get_paths_and_pattern(task, run)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        return

    # Collect task fMRI files: *_tfMRI_{TASK}_{run}.nii.gz
    task_files = []
    for p in images_dir.iterdir():
        if not p.is_file():
            continue
        m = task_fmri_pattern.match(p.name)
        if m:
            sub_id = m.group(1)
            task_files.append((sub_id, p))

    task_files.sort(key=lambda x: x[0])
    task_label = task if task == "WM" else task.upper()
    print(f"Found {len(task_files)} task fMRI files ({task_label}) in {images_dir}")

    # Process one atlas at a time: create masker once, then transform each file.
    # This avoids re-loading the atlas and re-fitting 1000+ times (was causing the "stuck").
    for atlas_name, atlas_path in atlases.items():
        if not atlas_path.exists():
            print(f"Atlas file not found: {atlas_path}")
            continue

        print(f"Loading atlas {atlas_name} (one-time)...")
        masker = NiftiLabelsMasker(
            labels_img=str(atlas_path),
            detrend=False,
            standardize=True,
        )
        # Fit on first file so we only do transform() for the rest
        first_sub_id, first_fmri = task_files[0]
        tqdm.write(f"  Fitting masker on first file ({first_sub_id})...")
        masker.fit(str(first_fmri))

        for sub_id, fmri_file in tqdm(
            task_files,
            desc=f"Atlas {atlas_name}",
            unit="file",
            file=sys.stdout,
        ):
            ts_out = output_dir / sub_id
            ts_out.mkdir(parents=True, exist_ok=True)
            out_ts = ts_out / f"{sub_id}_{atlas_name}_ts.npy"
            if out_ts.exists():
                continue
            try:
                ts, cols = extract_timeseries_with_masker(masker, fmri_file)
                np.save(out_ts, ts)
                np.save(ts_out / f"{sub_id}_{atlas_name}_valid_cols.npy", cols)
            except Exception as e:
                tqdm.write(f"Failed {sub_id} {atlas_name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess HCP task fMRI to ROI time series.")
    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        choices=TASK_NAMES,
        help="Task name (used in paths and filename pattern).",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=DEFAULT_RUN,
        choices=("LR", "RL"),
        help="Run subfolder under images/ (LR or RL).",
    )
    args = parser.parse_args()
    main(task=args.task, run=args.run)
