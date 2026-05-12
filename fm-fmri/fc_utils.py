"""
Functional connectivity (FC) utilities for cFID-FC.

- FC(x): V×V correlation matrix from time series x (T×V).
- vec(FC): flatten upper triangle (excluding diagonal) → d = V(V-1)/2.
- cFID-FC: Fréchet distance between distributions of f(real) and f(generated)
  on paired samples (same rest conditioning), with regularized covariances.

Cost for V ROIs: d = V(V-1)/2. With V=166, d ≈ 13,695. The implementation uses
covariances C_r, C_g of shape (d,d), then C_r @ C_g and sqrtm(·), so time is
O(N·d²) for cov + O(d³) for product and sqrt — with 166 ROIs this can be slow
and memory-heavy (~3 GB for the covariances). Use max_fc_dim to compute cFID
on a random projection to lower dimension for a faster approximate metric.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm


def correlation_matrix(x: np.ndarray) -> np.ndarray:
    """
    Compute sample correlation matrix of x.
    x: (T, V) time series.
    Returns: (V, V) correlation matrix. Diagonal = 1.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x must be (T, V)")
    T, V = x.shape
    # Center
    x_centered = x - x.mean(axis=0)
    # Covariance
    cov = (x_centered.T @ x_centered) / max(T - 1, 1)
    std = np.sqrt(np.diag(cov))
    std[std == 0] = 1.0
    # Correlation
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    return corr


def fc_upper_triangle_vector(fc: np.ndarray) -> np.ndarray:
    """
    Flatten upper triangle of FC matrix (excluding diagonal).
    fc: (V, V).
    Returns: (d,) with d = V(V-1)/2.
    """
    V = fc.shape[0]
    triu_inds = np.triu_indices(V, k=1)
    return fc[triu_inds].ravel().astype(np.float64)


def time_series_to_fc_vector(x: np.ndarray) -> np.ndarray:
    """
    f(x) = vec(FC(x)): correlation matrix then upper triangle.
    x: (T, V) time series.
    Returns: (d,) with d = V(V-1)/2.
    """
    fc = correlation_matrix(x)
    return fc_upper_triangle_vector(fc)


def fc_vectors_from_time_series_batch(X: np.ndarray) -> np.ndarray:
    """
    Compute FC feature vectors for a batch of time series.
    X: (N, T, V).
    Returns: (N, d) with d = V(V-1)/2.
    """
    N, T, V = X.shape
    d = V * (V - 1) // 2
    F = np.empty((N, d), dtype=np.float64)
    for i in range(N):
        F[i] = time_series_to_fc_vector(X[i])
    return F


def cfid_fc(
    X_real: np.ndarray,
    X_gen: np.ndarray,
    eps: float = 1e-6,
    max_fc_dim: int | None = None,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Conditional Fréchet distance using FC features (cFID-FC).

    Uses *paired* samples: X_real[i] and X_gen[i] share the same rest
    conditioning. Statistics are computed over {f(x_real_i)} and {f(x_gen_i)},
    then the Fréchet formula is applied.

    Formula:
        cFID_FC = ||μ_r - μ_g||^2 + Tr(C_r + C_g - 2 (C_r C_g)^{1/2})
    with μ_r, μ_g = means of f(real), f(generated);
    C_r, C_g = covariances (regularized: C + eps*I).

    Lower is better (better match of FC distribution).

    Parameters
    ----------
    X_real : (N, T, V) real task time series
    X_gen  : (N, T, V) generated task time series (same N, same order = same rest)
    eps    : regularization for covariances to avoid singular matrix sqrt
    max_fc_dim : if set, project FC vectors to this many random dimensions before
                 computing cFID (faster when d = V(V-1)/2 is large, e.g. V=166 → d≈13695).
                 Same projection for real and gen; approximate but comparable across runs.
    rng    : optional numpy Generator for reproducible random projection when max_fc_dim is set

    Returns
    -------
    cFID_FC : float
    """
    X_real = np.asarray(X_real, dtype=np.float64)
    X_gen = np.asarray(X_gen, dtype=np.float64)
    if X_real.shape != X_gen.shape:
        raise ValueError("X_real and X_gen must have the same shape (N, T, V)")

    F_r = fc_vectors_from_time_series_batch(X_real)  # (N, d)
    F_g = fc_vectors_from_time_series_batch(X_gen)   # (N, d)

    d_full = F_r.shape[1]
    if max_fc_dim is not None and max_fc_dim > 0 and max_fc_dim < d_full:
        if rng is None:
            rng = np.random.default_rng()
        proj = rng.standard_normal((d_full, max_fc_dim)).astype(np.float64)
        proj /= np.linalg.norm(proj, axis=0, keepdims=True)  # (d_full, max_fc_dim)
        F_r = F_r @ proj   # (N, max_fc_dim)
        F_g = F_g @ proj   # (N, max_fc_dim)
        d = max_fc_dim
    else:
        d = d_full

    mu_r = F_r.mean(axis=0)
    mu_g = F_g.mean(axis=0)
    C_r = np.cov(F_r, rowvar=False)
    C_g = np.cov(F_g, rowvar=False)

    # Regularize for numerical stability
    C_r = C_r + eps * np.eye(d)
    C_g = C_g + eps * np.eye(d)

    diff = mu_r - mu_g
    term_mean = np.sum(diff ** 2)

    # Matrix square root of (C_r @ C_g) for Tr(C_r + C_g - 2 sqrt(C_r C_g))
    try:
        covmean = sqrtm(C_r @ C_g)
        if np.iscomplexobj(covmean):
            covmean = np.real(covmean)
        term_cov = np.trace(C_r) + np.trace(C_g) - 2 * np.trace(covmean)
    except Exception:
        term_cov = np.nan

    cfid = term_mean + term_cov
    return float(cfid)


def cfid_fc_subject_level(
    X_real: np.ndarray,
    X_gen: np.ndarray,
    subject_ids: list,
    eps: float = 1e-6,
    max_fc_dim: int | None = None,
    rng: np.random.Generator | None = None,
    min_windows: int = 2,
) -> dict:
    """
    Compute per-subject cFID-FC, then return mean and std across subjects.

    Each subject contributes multiple windows (from the sliding-window dataset).
    The Fréchet distance is computed independently per subject using that subject's
    windows only, so it reflects how well the model captures *that subject's* FC
    distribution rather than the population average.

    Subjects with fewer than `min_windows` windows are skipped (Fréchet distance
    requires at least 2 samples to estimate a covariance).

    Parameters
    ----------
    X_real       : (N, T, V) all real windows concatenated
    X_gen        : (N, T, V) all generated windows (same order)
    subject_ids  : list of length N, subject label for each window
    eps          : covariance regularisation (passed to cfid_fc)
    max_fc_dim   : random projection dimension (same projection applied to all
                   subjects so scores are on a comparable scale)
    rng          : numpy Generator for reproducible projection (default seed=0)
    min_windows  : minimum windows a subject needs to be included (default 2)

    Returns
    -------
    dict with keys:
        'per_subject'  : dict  {subject_id -> cFID float}
        'mean'         : float  mean cFID across subjects (lower = better)
        'std'          : float  std  cFID across subjects
        'n_subjects'   : int    number of subjects successfully scored
        'n_skipped'    : int    subjects skipped (too few windows or sqrtm failed)
    """
    from collections import defaultdict

    X_real = np.asarray(X_real, dtype=np.float64)
    X_gen  = np.asarray(X_gen,  dtype=np.float64)
    if X_real.shape != X_gen.shape:
        raise ValueError("X_real and X_gen must have the same shape (N, T, V)")
    if len(subject_ids) != X_real.shape[0]:
        raise ValueError("len(subject_ids) must equal X_real.shape[0]")

    # Build a shared random projection once (same matrix for all subjects so
    # scores are on a comparable scale and can be averaged meaningfully).
    proj = None
    if max_fc_dim is not None and max_fc_dim > 0:
        # Determine full FC dimension from one sample
        d_full = len(time_series_to_fc_vector(X_real[0]))
        if max_fc_dim < d_full:
            if rng is None:
                rng = np.random.default_rng(0)
            proj = rng.standard_normal((d_full, max_fc_dim)).astype(np.float64)
            proj /= np.linalg.norm(proj, axis=0, keepdims=True)

    # Group window indices by subject
    subj_indices: dict = defaultdict(list)
    for i, sid in enumerate(subject_ids):
        subj_indices[sid].append(i)

    per_subject: dict = {}
    n_skipped = 0

    for sid, idxs in sorted(subj_indices.items()):
        if len(idxs) < min_windows:
            n_skipped += 1
            continue

        Fr = fc_vectors_from_time_series_batch(X_real[idxs])  # (n_w, d_full)
        Fg = fc_vectors_from_time_series_batch(X_gen[idxs])

        # Apply shared projection if set
        if proj is not None:
            Fr = Fr @ proj   # (n_w, max_fc_dim)
            Fg = Fg @ proj

        d = Fr.shape[1]
        mu_r = Fr.mean(axis=0)
        mu_g = Fg.mean(axis=0)
        C_r = np.cov(Fr, rowvar=False) + eps * np.eye(d)
        C_g = np.cov(Fg, rowvar=False) + eps * np.eye(d)

        diff      = mu_r - mu_g
        term_mean = float(np.sum(diff ** 2))
        try:
            covmean = sqrtm(C_r @ C_g)
            if np.iscomplexobj(covmean):
                covmean = np.real(covmean)
            term_cov = float(np.trace(C_r) + np.trace(C_g) - 2.0 * np.trace(covmean))
        except Exception:
            n_skipped += 1
            continue

        score = term_mean + term_cov
        if not np.isnan(score):
            per_subject[sid] = score
        else:
            n_skipped += 1

    scores = list(per_subject.values())
    return {
        "per_subject": per_subject,
        "mean":        float(np.mean(scores)) if scores else float("nan"),
        "std":         float(np.std(scores))  if scores else float("nan"),
        "n_subjects":  len(per_subject),
        "n_skipped":   n_skipped,
    }


def _fc_upper_triangle_flat(fc: np.ndarray) -> np.ndarray:
    """Upper triangle (k=1) of FC as 1D array. (V,V) -> (d,) with d = V*(V-1)/2."""
    mask = np.triu(np.ones_like(fc, dtype=bool), k=1)
    return np.asarray(fc[mask], dtype=np.float64)


def compute_fc_precision_at_5_single(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Precision@5%: of predicted top 5% edges (by |FC_pred|), fraction that are in ground-truth top 5%.
    pred, target: (T, V) time series. Returns scalar in [0, 1].
    """
    fc_pred = correlation_matrix(pred)
    fc_tgt = correlation_matrix(target)
    pred_flat = _fc_upper_triangle_flat(fc_pred)
    gt_flat = _fc_upper_triangle_flat(fc_tgt)
    N = len(pred_flat)
    if N == 0:
        return float("nan")
    n_top = max(1, int(round(N * 5.0 / 100.0)))
    pred_abs = np.abs(pred_flat)
    gt_abs = np.abs(gt_flat)
    pred_order = np.argsort(-pred_abs)
    gt_order = np.argsort(-gt_abs)
    gt_top_idx = set(gt_order[:n_top])
    pred_top_idx = set(pred_order[:n_top])
    inter = len(gt_top_idx & pred_top_idx)
    return float(inter / len(pred_top_idx)) if pred_top_idx else 0.0


def compute_fc_precision_at_5_paired(X_real: np.ndarray, X_gen: np.ndarray) -> float:
    """
    Window-level average of FC precision@5%.
    X_real, X_gen: (N, T, V). Returns mean over windows (nan for invalid windows excluded).
    """
    X_real = np.asarray(X_real, dtype=np.float64)
    X_gen = np.asarray(X_gen, dtype=np.float64)
    if X_real.shape != X_gen.shape:
        raise ValueError("X_real and X_gen must have the same shape (N, T, V)")
    N = X_real.shape[0]
    prec_list = []
    for i in range(N):
        try:
            p = compute_fc_precision_at_5_single(X_gen[i], X_real[i])
            if not np.isnan(p):
                prec_list.append(p)
        except Exception:
            pass
    return float(np.mean(prec_list)) if prec_list else float("nan")
