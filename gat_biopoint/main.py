"""
GAT on Biopoint: autism classification from AAL ROI time series.
Uses GAT (STNAGNN-fMRI style) for classification; supports fMRI synthetic data augmentation.
Usage:
  python main.py --train [-ds /path/to/biopoint_data] [-dt /path/to/results]
  python main.py --test  [-ds ...] [-dt ...]
  python main.py --train --test
  python main.py --train --test --use_synthetic --synthetic_dir /path/to/synthetic_biopoint
  python main.py --compare --use_synthetic --synthetic_dir /path/to/synthetic   # real vs synthetic head-to-head
"""

from util.option import parse
from experiment import train, test, compare


if __name__ == "__main__":
    argv = parse()
    if argv.compare:
        if not getattr(argv, "use_synthetic", False):
            raise ValueError("--compare requires --use_synthetic --synthetic_dir <path>")
        compare(argv)
    else:
        if not argv.train and not argv.test:
            argv.train = True
            argv.test = True
        if argv.train:
            train(argv)
        if argv.test:
            test(argv)
