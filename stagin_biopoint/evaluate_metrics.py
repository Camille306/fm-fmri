"""
Post-hoc evaluation script for stagin_biopoint.
Loads saved samples.pkl and reports F1, sensitivity, specificity, accuracy, AUC.
Usage:
    python evaluate_metrics.py --result_dir ./result/stagin_biopoint
"""

import argparse
import os
import numpy as np
import torch
from sklearn import metrics


def compute_metrics(true, pred, prob):
    accuracy    = metrics.accuracy_score(true, pred)
    f1          = metrics.f1_score(true, pred, average="binary")
    sensitivity = metrics.recall_score(true, pred, average="binary")   # TP / (TP + FN)
    roc_auc     = metrics.roc_auc_score(true, prob[:, 1])

    tn, fp, fn, tp = metrics.confusion_matrix(true, pred).ravel()
    specificity = tn / (tn + fp)                                        # TN / (TN + FP)

    return dict(
        accuracy=accuracy,
        f1=f1,
        sensitivity=sensitivity,
        specificity=specificity,
        roc_auc=roc_auc,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True, help="Path to result directory (contains samples.pkl)")
    args = parser.parse_args()

    pkl_path = os.path.join(args.result_dir, "samples.pkl")
    assert os.path.isfile(pkl_path), f"samples.pkl not found at {pkl_path}"

    samples = torch.load(pkl_path)
    true = samples["true"]
    pred = samples["pred"]
    prob = samples["prob"]

    spacer = 15

    # Per-fold results
    if isinstance(true, dict):
        folds = sorted(true.keys())
        fold_metrics = []
        print("\n=== Per-Fold Results ===")
        for k in folds:
            m = compute_metrics(true[k], pred[k], prob[k])
            fold_metrics.append(m)
            print(f"\nFold {k}:")
            for key, val in m.items():
                print(f"  {key:{spacer}}: {val:.4f}")

        print("\n=== Aggregated Results ===")
        keys = fold_metrics[0].keys()
        mean_m = {k: np.mean([m[k] for m in fold_metrics]) for k in keys}
        std_m  = {k: np.std( [m[k] for m in fold_metrics]) for k in keys}
        for k in keys:
            print(f"  {k:{spacer}}: {mean_m[k]:.4f} ± {std_m[k]:.4f}")

    else:
        # Single split
        m = compute_metrics(true, pred, prob)
        print("\n=== Results ===")
        for key, val in m.items():
            print(f"  {key:{spacer}}: {val:.4f}")


if __name__ == "__main__":
    main()
