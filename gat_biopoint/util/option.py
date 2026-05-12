# GAT Biopoint options

import os
import csv
import argparse


def parse():
    parser = argparse.ArgumentParser(description="GAT on Biopoint (autism classification, fMRI synthetic data augmentation)")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-n", "--exp_name", type=str, default="gat_biopoint")
    parser.add_argument("-k", "--k_fold", type=int, default=5, help="CV folds; 1 = single train/test split using --train_ratio")
    parser.add_argument("--fold", type=int, default=None, help="Train/test only this fold (0-indexed). If None, run all folds.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Fraction for training when k_fold=1; stratified split")
    parser.add_argument("-b", "--minibatch_size", type=int, default=10)

    parser.add_argument("-ds", "--sourcedir", type=str, default="./data/biopoint_data", help="Biopoint data root")
    parser.add_argument("-dt", "--targetdir", type=str, default="./result")
    parser.add_argument("--csv_path", type=str, default="./data/biopoint_data.csv")

    # Atlas selection:
    # - shen268: legacy `.npy` in sourcedir/output/<id>/{rest,task}/<id>_shen268_ts.npy
    # - dk:      `.pt` in dk_atlas_ts_root/<id>_{rest,task}_roi_ts.pt
    parser.add_argument("--atlas_source", type=str, default="shen268", choices=["shen268", "dk"])
    parser.add_argument("--dk_atlas_ts_root", type=str, default="./data/biopoint_dk_atlas")
    parser.add_argument("--ts_filename_suffix", type=str, default="_shen268_ts.npy",
                        help="ROI file suffix when atlas_source=shen268 (default: _shen268_ts.npy)")
    parser.add_argument("--data_type", type=str, default="rest", choices=["rest", "task"],
                        help="Use resting-state ('rest') or task ('task') ROI time series")
    parser.add_argument("--dynamic_length", type=int, default=None)
    parser.add_argument("--use_synthetic", action="store_true", help="Use real + synthetic (fm-fmri) for training")
    parser.add_argument("--synthetic_dir", type=str, default="./synthetic_biopoint")

    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--window_stride", type=int, default=3)
    parser.add_argument("--window_num", type=int, default=12, help="Number of dynamic FC snapshots per subject")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=15, help="Training epochs per fold (no early stopping)")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--model", type=str, default="v1", choices=["v1", "v2"],
                        help="v1 = original model.py, v2 = configurable model_v2.py")
    parser.add_argument("--num_heads", type=int, default=1,
                        help="GAT attention heads (v2 only)")
    parser.add_argument("--use_gatv2", action="store_true",
                        help="Use GATv2Conv instead of GATConv (v2 only)")
    parser.add_argument("--num_gat_layers", type=int, default=2, choices=[2, 3],
                        help="Number of stacked GAT layers (v2 only)")
    parser.add_argument("--residual", action="store_true",
                        help="Residual connection between GAT layers (v2 only)")
    parser.add_argument("--mlp_hidden", type=int, default=0,
                        help="FC1 width (0 = hidden_dim*4). v2 only.")
    parser.add_argument("--mlp_bottleneck", type=int, default=32,
                        help="FC2 width before classifier (v2 only)")
    parser.add_argument("--pool_mode", type=str, default="mean_max",
                        choices=["mean_max", "mean", "max", "attention"],
                        help="Per-snapshot pooling strategy (v2 only)")
    parser.add_argument("--sparsity", type=int, default=0,
                        help="FC edge sparsity percentile (0=dense, 30=keep top 70%%)")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["step", "cosine"],
                        help="LR scheduler: cosine (CosineAnnealing) or step (StepLR)")
    parser.add_argument("--step_size", type=int, default=10, help="StepLR: decay lr every N epochs")
    parser.add_argument("--step_gamma", type=float, default=0.4, help="StepLR: multiply lr by gamma")
    parser.add_argument("--noise_level", type=float, default=0.05,
                        help="Gaussian noise std added to time series during training (0=off)")
    parser.add_argument("--bootstrap_n", type=int, default=0,
                        help="Bootstrap augmentation: replicate each real training subject N times "
                             "with randomly resampled temporal windows (0=off, paper uses 30)")

    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device: cuda (GPU) or cpu")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--compare", action="store_true",
                        help="Head-to-head: train real-only AND real+synthetic on same folds, compare on same test set")

    argv = parser.parse_args()
    argv.targetdir = os.path.join(argv.targetdir, argv.exp_name)
    os.makedirs(argv.targetdir, exist_ok=True)
    with open(os.path.join(argv.targetdir, "argv.csv"), "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(vars(argv).items())
    return argv
