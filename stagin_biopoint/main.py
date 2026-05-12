"""
STAGIN on Biopoint: autism classification from AAL ROI time series.
Usage:
  python main.py --train [-ds /path/to/biopoint_data] [-dt /path/to/results]
  python main.py --test  [-ds ...] [-dt ...]
  python main.py --train --test
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
