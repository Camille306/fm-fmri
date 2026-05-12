"""
GCN/GAT on Biopoint: autism binary classification from resting-state ROI time series.
Supports --model_type gat|gcn and real-only or real+synthetic augmentation.

Usage:
  # Real-only, GAT, 5-fold CV
  python main.py --train --test --model_type gat -dt ./results/gat_real

  # Real+synthetic, GAT, 5-fold CV
  python main.py --train --test --model_type gat --use_synthetic \
    --synthetic_dir /path/to/synthetic_biopoint -dt ./results/gat_synthetic

  # Real-only, GCN
  python main.py --train --test --model_type gcn -dt ./results/gcn_real

  # Real+synthetic, GCN
  python main.py --train --test --model_type gcn --use_synthetic \
    --synthetic_dir /path/to/synthetic_biopoint -dt ./results/gcn_synthetic
"""

import argparse
from util.option import parse
from experiment import train, test


if __name__ == "__main__":
    argv = parse()
    if not argv.train and not argv.test:
        argv.train = True
        argv.test = True
    if argv.train:
        train(argv)
    if argv.test:
        test(argv)
