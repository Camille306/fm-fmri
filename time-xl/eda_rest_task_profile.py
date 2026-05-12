"""
Load 3 examples from REST and 3 examples from TASK directly from disk (no Dataset class).

This is meant for quick profiling / sanity checks:
- which exact files get loaded
- array shapes (timepoints, variables)
- dtype and basic stats

Usage (example):
  python load_3_examples_rest_task.py ^
    --data_root "/path/to/hcp-resting-fc" ^
    --task_root "/path/to/hcp-task-ts" ^
    --task_name emotion ^
    --output_dir "./example_loads"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


DEFAULT_DATA_ROOT = "./data/hcp-resting-fc"
DEFAULT_TASK_ROOT = "./data/hcp-task-ts"


def rest_path(data_root: Path, subject_id: str) -> Path:
    # Use AAL3 parcellation to match task ROI dimensionality.
    return data_root / subject_id / "timeseries" / "REST1_LR_AAL3_ts.npy"


def find_task_path(task_root: Path, task_name: str, subject_id: str) -> Optional[Path]:
    """
    Replicates the path heuristics from dataset.py, but without importing it.
    """
    parent_dir = task_root / task_name / "roi_data_std"
    base = parent_dir / subject_id

    # Pattern 1: file without extension
    if base.exists() and base.is_file():
        return base

    # Pattern 2: directory containing file(s)
    if base.exists() and base.is_dir():
        for ext in [".pt", ".pth", ".npy"]:
            cand = base / f"{subject_id}{ext}"
            if cand.exists():
                return cand
        cand = base / subject_id
        if cand.exists():
            return cand

    # Pattern 3: file with extension in parent dir
    for ext in [".pt", ".pth", ".npy"]:
        cand = parent_dir / f"{subject_id}{ext}"
        if cand.exists():
            return cand

    # Pattern 4: file without extension in parent dir
    cand = parent_dir / subject_id
    if cand.exists():
        return cand

    return None


def load_task_file(path: Path) -> np.ndarray:
    """
    Load task file from .pt/.pth (torch) or .npy (numpy).
    """
    suffix = path.suffix.lower()

    if suffix in [".npy"]:
        return np.load(str(path))

    # Try torch load first for .pt/.pth or no suffix
    try:
        import torch  # local import so script can still run if only .npy is used

        obj = torch.load(str(path), map_location="cpu")
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
        if isinstance(obj, np.ndarray):
            return obj
        if isinstance(obj, dict):
            if "data" in obj:
                data = obj["data"]
                if isinstance(data, torch.Tensor):
                    return data.detach().cpu().numpy()
                if isinstance(data, np.ndarray):
                    return data
            # fallback: first tensor/array value
            for v in obj.values():
                if isinstance(v, torch.Tensor):
                    return v.detach().cpu().numpy()
                if isinstance(v, np.ndarray):
                    return v
            raise ValueError(f"Could not extract tensor/ndarray from dict keys={list(obj.keys())}")
        raise ValueError(f"Unsupported torch-loaded object type: {type(obj)}")
    except Exception:
        # fallback to numpy
        return np.load(str(path))


def as_2d_time_by_var(arr: np.ndarray, expected_vars: Optional[int] = None) -> Tuple[np.ndarray, str]:
    """
    Normalize to (T, V) for reporting.
    Returns: (arr2d, note)
    """
    note_parts = []
    x = np.asarray(arr)
    orig_shape = tuple(x.shape)

    x = np.squeeze(x)
    if tuple(x.shape) != orig_shape:
        note_parts.append(f"squeezed {orig_shape}->{tuple(x.shape)}")

    if x.ndim == 1:
        return x.reshape(-1, 1), " ; ".join(note_parts + ["1D->(T,1)"])

    if x.ndim == 2:
        if expected_vars is not None and x.shape[0] == expected_vars and x.shape[1] != expected_vars:
            return x.T, " ; ".join(note_parts + [f"transposed because first_dim==expected_vars({expected_vars})"])
        if expected_vars is None and x.shape[0] == 268 and x.shape[1] != 268:
            return x.T, " ; ".join(note_parts + ["transposed because first_dim==268"])
        return x, " ; ".join(note_parts)

    # Fallback: flatten everything except first dim
    x2 = x.reshape(x.shape[0], -1)
    note_parts.append(f"flattened {tuple(x.shape)}-> {tuple(x2.shape)}")
    return x2, " ; ".join(note_parts)


def summarize_array(name: str, arr: np.ndarray) -> str:
    arr = np.asarray(arr)
    finite = np.isfinite(arr)
    finite_vals = arr[finite] if finite.any() else np.array([], dtype=float)

    lines = []
    lines.append(f"{name}:")
    lines.append(f"  dtype={arr.dtype}, shape={arr.shape}")
    if finite_vals.size:
        lines.append(
            "  min={:.6g} max={:.6g} mean={:.6g} std={:.6g}".format(
                float(finite_vals.min()),
                float(finite_vals.max()),
                float(finite_vals.mean()),
                float(finite_vals.std()),
            )
        )
        lines.append(f"  finite={finite_vals.size}/{arr.size} ({finite_vals.size/arr.size:.2%})")
    else:
        lines.append("  (no finite values)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Directly load 3 rest + 3 task examples (no Dataset class).")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--task_root", type=str, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--task_name", type=str, default="emotion")
    parser.add_argument("--n_examples", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="./example_loads")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    task_root = Path(args.task_root)
    os.makedirs(args.output_dir, exist_ok=True)

    # Find candidate subject IDs from data_root directories
    subject_ids = sorted([p.name for p in data_root.iterdir() if p.is_dir()])

    picked = []
    for sid in subject_ids:
        rp = rest_path(data_root, sid)
        tp = find_task_path(task_root, args.task_name, sid)
        if rp.exists() and tp is not None and tp.exists():
            picked.append((sid, rp, tp))
        if len(picked) >= args.n_examples:
            break

    out_lines = []
    out_lines.append("Direct load examples (no Dataset class)")
    out_lines.append("=" * 72)
    out_lines.append(f"data_root={data_root}")
    out_lines.append(f"task_root={task_root}")
    out_lines.append(f"task_name={args.task_name}")
    out_lines.append(f"n_examples={args.n_examples}")
    out_lines.append("")

    if not picked:
        out_lines.append("No paired subjects found (rest file + task file).")
        report_path = Path(args.output_dir) / "examples_report.txt"
        report_path.write_text("\n".join(out_lines), encoding="utf-8")
        print("\n".join(out_lines))
        print(f"\nSaved report to: {report_path}")
        return

    for i, (sid, rp, tp) in enumerate(picked, start=1):
        out_lines.append(f"[{i}] subject_id={sid}")
        out_lines.append(f"  rest_path={rp}")
        out_lines.append(f"  task_path={tp}")

        # Load rest
        rest = np.load(str(rp))
        rest2d, rest_note = as_2d_time_by_var(rest, expected_vars=268 if rest.ndim >= 2 else None)
        out_lines.append(summarize_array("  REST raw", rest))
        if rest_note:
            out_lines.append(f"  REST note: {rest_note}")
        out_lines.append(summarize_array("  REST (T,V)", rest2d))

        # Load task
        task = load_task_file(tp)
        task2d, task_note = as_2d_time_by_var(task, expected_vars=rest2d.shape[1])
        out_lines.append(summarize_array("  TASK raw", task))
        if task_note:
            out_lines.append(f"  TASK note: {task_note}")
        out_lines.append(summarize_array("  TASK (T,V)", task2d))

        out_lines.append("")  # spacer

    report_path = Path(args.output_dir) / "examples_report.txt"
    report_path.write_text("\n".join(out_lines), encoding="utf-8")
    print("\n".join(out_lines))
    print(f"\nSaved report to: {report_path}")


if __name__ == "__main__":
    main()

