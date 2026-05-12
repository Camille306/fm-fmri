# GCN/GAT Biopoint options

import os
import csv
import argparse


def parse():
    parser = argparse.ArgumentParser(
        description="GCN/GAT on Biopoint (autism binary classification, fMRI synthetic data augmentation)"
    )
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-n", "--exp_name", type=str, default="gcn_gat_biopoint")
    parser.add_argument(
        "--model_type", type=str, default="gat", choices=["gat", "gcn"],
        help="Graph model: 'gat' (Graph Attention Network) or 'gcn' (Graph Convolutional Network)"
    )
    parser.add_argument(
        "-k", "--k_fold", type=int, default=5,
        help="CV folds; 1 = single train/test split using --train_ratio"
    )
    parser.add_argument(
        "--max_folds", type=int, default=1,
        help="How many folds to actually run (1 = only fold 0, fastest). "
             "Set to k_fold to run full CV."
    )
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Fraction for training when k_fold=1; stratified split")
    parser.add_argument("-b", "--minibatch_size", type=int, default=4,
                        help="Batch size. Keep small (2-4) for 268-ROI graphs to avoid OOM")

    parser.add_argument("-ds", "--sourcedir", type=str,
                        default="./data/biopoint_data",
                        help="Biopoint data root")
    parser.add_argument("-dt", "--targetdir", type=str, default="./result")
    parser.add_argument("--csv_path", type=str,
                        default="./data/biopoint_data.csv")
    # Atlas selection:
    # - shen268: legacy `.npy` in sourcedir/output/<id>/{rest,task}/<id>_shen268_ts.npy
    # - dk:      `.pt` in dk_atlas_ts_root/<id>_{rest,task}_roi_ts.pt
    parser.add_argument("--atlas_source", type=str, default="shen268", choices=["shen268", "dk"])
    parser.add_argument("--dk_atlas_ts_root", type=str, default="./data/biopoint_dk_atlas")
    parser.add_argument("--ts_filename_suffix", type=str, default="_shen268_ts.npy",
                        help="ROI file suffix when atlas_source=shen268 (default: _shen268_ts.npy)")
    parser.add_argument("--dynamic_length", type=int, default=None)
    parser.add_argument("--use_synthetic", action="store_true",
                        help="Use real + synthetic (fm-fmri) for training")
    parser.add_argument("--synthetic_dir", type=str,
                        default="./synthetic_biopoint")
    parser.add_argument("--use_task", action="store_true",
                        help="Use task fMRI instead of resting-state fMRI")

    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--window_stride", type=int, default=3)
    parser.add_argument("--window_num", type=int, default=12,
                        help="Number of dynamic FC snapshots per subject")
    parser.add_argument("--top_k_edges", type=int, default=50,
                        help="Keep top-K strongest |FC| neighbours per node per snapshot. "
                             "Reduces edges from 268*267=71k to 268*K per snapshot. "
                             "Lower values save memory; 30-60 is a good range.")

    # -----------------------------------------------------------------------
    # Quality filtering / curriculum learning
    # -----------------------------------------------------------------------
    parser.add_argument(
        "--quality_frac", type=float, default=1.0,
        help=(
            "Fraction of training subjects to keep, ranked by FC signal quality "
            "(mean |off-diagonal FC| of full resting-state scan). "
            "1.0 = use all subjects (default). "
            "0.7 = discard the noisiest 30%% of subjects. "
            "Combine with --curriculum to ramp from quality_frac→1.0 over training."
        ),
    )
    parser.add_argument(
        "--curriculum", action="store_true",
        help=(
            "Enable curriculum learning: start training with quality_frac of subjects "
            "(cleanest), linearly expanding to 100%% by the final epoch. "
            "Requires --quality_frac < 1.0 to have any effect."
        ),
    )

    # -----------------------------------------------------------------------
    # Model / optimiser hyperparameters  ← things to tune
    # -----------------------------------------------------------------------
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate. Try: 5e-5, 1e-4, 3e-4")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Adam weight decay (L2). Try: 1e-4, 1e-3, 0.01")
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Training epochs. Try: 30, 50, 100")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="GNN hidden dimension. Try: 64, 128, 256")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate. Try: 0.1, 0.2, 0.5")
    parser.add_argument("--scheduler_step", type=int, default=10,
                        help="StepLR step size (epochs). Try: 5, 10, 20")
    parser.add_argument("--scheduler_gamma", type=float, default=0.4,
                        help="StepLR decay factor. Try: 0.5, 0.4, 0.1")

    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"],
                        help="Device: cuda (GPU) or cpu")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")

    argv = parser.parse_args()
    # Incorporate model_type into exp_name so real-only and synthetic runs get separate dirs
    argv.targetdir = os.path.join(argv.targetdir, argv.exp_name)
    os.makedirs(argv.targetdir, exist_ok=True)
    with open(os.path.join(argv.targetdir, "argv.csv"), "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(vars(argv).items())
    return argv
