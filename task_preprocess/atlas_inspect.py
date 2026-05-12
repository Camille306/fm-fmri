#!/usr/bin/env python3
"""
Count unique values (labels/ROIs) in an atlas NIfTI image.
Usage: python count_atlas_labels.py <atlas.nii.gz>
       python count_atlas_labels.py  (uses default atlas paths)
"""

import sys
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError:
    print("Error: nibabel is required. Install with: pip install nibabel")
    sys.exit(1)


def count_unique_labels(atlas_path: Path) -> dict:
    """
    Load a NIfTI atlas and count unique values (excluding NaN and optionally 0).
    
    Args:
        atlas_path: Path to .nii or .nii.gz atlas file
        
    Returns:
        dict with keys: n_unique, n_unique_nonzero, unique_values, shape, dtype
    """
    atlas_path = Path(atlas_path)
    if not atlas_path.exists():
        raise FileNotFoundError(f"Atlas file not found: {atlas_path}")
    
    img = nib.load(str(atlas_path))
    data = np.asarray(img.dataobj)  # avoid loading into memory if not needed
    
    # Flatten and get unique values
    flat = data.ravel()
    
    # Exclude NaN for float atlases
    finite = flat[np.isfinite(flat)]
    unique_all = np.unique(finite)
    
    # Non-zero (0 often means "no label" in atlases)
    nonzero = unique_all[unique_all != 0]
    
    return {
        "n_unique": len(unique_all),
        "n_unique_nonzero": len(nonzero),
        "unique_values": unique_all,
        "unique_nonzero": nonzero,
        "shape": data.shape,
        "dtype": str(data.dtype),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
    }


def main():
    atlas_dir = "./data/data"
    default_atlases = [
        atlas_dir + '/' + "AAL3v1.nii.gz",
    ]
    
    if len(sys.argv) > 1:
        paths = [Path(p) for p in sys.argv[1:]]
    else:
        paths = default_atlases
    
    for atlas_path in paths:
        print(f"\nAtlas: {atlas_path}")
        print("-" * 50)
        try:
            info = count_unique_labels(atlas_path)
            print(f"  Shape:           {info['shape']}")
            print(f"  Dtype:            {info['dtype']}")
            print(f"  Min value:       {info['min']}")
            print(f"  Max value:       {info['max']}")
            print(f"  Unique values:   {info['n_unique']}")
            print(f"  Unique (non‑zero): {info['n_unique_nonzero']}")
            if info["n_unique"] <= 50:
                print(f"  Values:          {info['unique_values'].tolist()}")
            else:
                print(f"  First 20 values: {info['unique_values'][:20].tolist()} ...")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    main()
