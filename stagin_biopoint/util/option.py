# STAGIN options + biopoint dataset

import os
import csv
import argparse


def parse():
    parser = argparse.ArgumentParser(description="STAGIN on Biopoint (autism classification)")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-n", "--exp_name", type=str, default="stagin_biopoint")
    parser.add_argument("-k", "--k_fold", type=int, default=1, help="CV folds; 1 = single train/test split using --train_ratio")
    parser.add_argument("--fold", type=int, default=None, help="Train/test only this fold (0-indexed). If None, run all folds.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Fraction for training when k_fold=1 (test = 1 - train_ratio); stratified split")
    parser.add_argument("-b", "--minibatch_size", type=int, default=8)

    parser.add_argument("-ds", "--sourcedir", type=str, default="./data/biopoint_data", help="Biopoint data root (contains output/<subject_id>/rest/<id>_*_ts.npy)")
    parser.add_argument("-dt", "--targetdir", type=str, default="./result")
    parser.add_argument("--csv_path", type=str, default="./data/biopoint_data.csv", help="Path to biopoint CSV (subject_id, group). Default: project path")

    parser.add_argument("--dataset", type=str, default="biopoint-rest", choices=["biopoint-rest"])
    # Atlas selection:
    # - shen268: legacy `.npy` in sourcedir/output/<id>/{rest,task}/<id>_shen268_ts.npy
    # - dk:      `.pt` in dk_atlas_ts_root/<id>_{rest,task}_roi_ts.pt
    parser.add_argument("--atlas_source", type=str, default="shen268", choices=["shen268", "dk"])
    parser.add_argument("--dk_atlas_ts_root", type=str, default="./data/biopoint_dk_atlas")
    parser.add_argument("--ts_filename_suffix", type=str, default="_shen268_ts.npy", help="ROI file suffix when atlas_source=shen268 (default: _shen268_ts.npy)")
    parser.add_argument("--dynamic_length", type=int, default=None, help="Fixed length window (default: use full time series)")
    parser.add_argument("--use_synthetic", action="store_true", help="Use real + synthetic (fm-fmri) training data")
    parser.add_argument("--synthetic_dir", type=str, default="./synthetic_biopoint", help="Directory with *_syn.npy and synthetic_manifest.csv (for --use_synthetic)")

    parser.add_argument("--quality_frac", type=float, default=1.0,
                        help="Keep top N%% of synthetic by FC quality (1.0=all, 0.5=top 50%%)")

    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--window_stride", type=int, default=3)

    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--max_lr", type=float, default=0.001)
    parser.add_argument("--reg_lambda", type=float, default=0.00001)
    parser.add_argument("--clip_grad", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--sparsity", type=int, default=30)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--readout", type=str, default="sero", choices=["garo", "sero", "mean"])
    parser.add_argument("--cls_token", type=str, default="sum", choices=["sum", "mean", "param"])

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--validate", action="store_true")

    parser.add_argument("--num_workers", type=int, default=0)

    argv = parser.parse_args()
    argv.targetdir = os.path.join(argv.targetdir, argv.exp_name)
    os.makedirs(argv.targetdir, exist_ok=True)
    with open(os.path.join(argv.targetdir, "argv.csv"), "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(vars(argv).items())
    return argv
