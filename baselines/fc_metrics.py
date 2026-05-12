"""
Shared FC metrics for baselines: functional connectivity, FC similarity, and top-k
precision/recall/AUC (filtering top 5%, 10%, 20%, 50% connectivities).
"""
import numpy as np
from scipy.stats import pearsonr


def compute_functional_connectivity(data: np.ndarray) -> np.ndarray:
    fc = np.corrcoef(data.T)
    return np.nan_to_num(fc, nan=0.0, posinf=1.0, neginf=-1.0)


def compute_fc_similarity(pred: np.ndarray, target: np.ndarray) -> float:
    fc_pred = compute_functional_connectivity(pred)
    fc_tgt = compute_functional_connectivity(target)
    mask = np.triu(np.ones_like(fc_pred, dtype=bool), k=1)
    a, b = fc_pred[mask], fc_tgt[mask]
    if len(a) > 1 and np.std(a) > 1e-10 and np.std(b) > 1e-10:
        r, _ = pearsonr(a, b)
        return float(r) if not np.isnan(r) else 0.0
    return 0.0


def _fc_upper_triangle_flat(fc: np.ndarray) -> np.ndarray:
    """Upper triangle (k=1) of FC as 1D array. (V,V) -> (N,) with N = V*(V-1)/2."""
    mask = np.triu(np.ones_like(fc, dtype=bool), k=1)
    return np.asarray(fc[mask], dtype=np.float64)


def compute_fc_topk_precision_recall_auc(
    pred: np.ndarray,
    target: np.ndarray,
    k_percentiles: tuple = (5, 10, 20, 50),
) -> dict:
    """
    For each k in k_percentiles (e.g. 5, 10, 20, 50):
    - Ground-truth top k% connectivities = edges with largest |FC_target|.
    - Predicted top k% = edges with largest |FC_pred|.
    - Precision@k = |intersection| / |pred_top_k|
    - Recall@k = |intersection| / |gt_top_k|
    - AUC@k = ROC-AUC with binary label = 1 if edge in GT top k%, score = |FC_pred|.

    Returns dict with keys like "precision_at_5", "recall_at_5", "auc_at_5", etc., and "k_percentiles".
    """
    fc_pred = compute_functional_connectivity(pred)
    fc_tgt = compute_functional_connectivity(target)
    pred_flat = _fc_upper_triangle_flat(fc_pred)
    gt_flat = _fc_upper_triangle_flat(fc_tgt)
    N = len(pred_flat)
    if N == 0:
        out = {}
        for k in k_percentiles:
            out[f"precision_at_{k}"] = out[f"recall_at_{k}"] = out[f"auc_at_{k}"] = float("nan")
        out["k_percentiles"] = k_percentiles
        return out

    pred_abs = np.abs(pred_flat)
    gt_abs = np.abs(gt_flat)
    pred_order = np.argsort(-pred_abs)
    gt_order = np.argsort(-gt_abs)

    out = {}
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        roc_auc_score = None

    for k_pct in k_percentiles:
        n_top = max(1, int(round(N * k_pct / 100.0)))
        gt_top_idx = set(gt_order[:n_top])
        pred_top_idx = set(pred_order[:n_top])
        inter = len(gt_top_idx & pred_top_idx)
        prec = inter / len(pred_top_idx) if pred_top_idx else 0.0
        rec = inter / len(gt_top_idx) if gt_top_idx else 0.0
        out[f"precision_at_{k_pct}"] = float(prec)
        out[f"recall_at_{k_pct}"] = float(rec)

        if roc_auc_score is not None:
            y_true = np.zeros(N, dtype=np.int32)
            y_true[list(gt_top_idx)] = 1
            y_score = pred_abs
            if np.unique(y_true).size == 2:
                auc_k = roc_auc_score(y_true, y_score)
                out[f"auc_at_{k_pct}"] = float(auc_k)
            else:
                out[f"auc_at_{k_pct}"] = float("nan")
        else:
            out[f"auc_at_{k_pct}"] = float("nan")

    out["k_percentiles"] = k_percentiles
    return out


# Default k percentiles for aggregation keys
TOP_K_PERCENTILES = (5, 10, 20, 50)


def topk_metric_keys():
    """Return list of metric keys for precision/recall/auc at each k."""
    return (
        [f"precision_at_{k}" for k in TOP_K_PERCENTILES]
        + [f"recall_at_{k}" for k in TOP_K_PERCENTILES]
        + [f"auc_at_{k}" for k in TOP_K_PERCENTILES]
    )
