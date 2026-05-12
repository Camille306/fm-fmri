"""
Biopoint dataset for GCN/GAT: load ROI time series and convert to PyG Data
(dynamic FC → graph per snapshot). Identical to gat_biopoint but model-agnostic.
Supports real-only or real + synthetic for data augmentation.

Quality filtering / curriculum learning
---------------------------------------
Synthetic samples from fm-fmri vary in quality — some generated time series
have weaker, noisier FC than others.  We score each synthetic sample by the
mean absolute off-diagonal FC of its time series and keep only the top fraction.

Real subjects are always kept in full (no quality filtering on real data).

Two modes for synthetic filtering (controlled from experiment.py):
  • Hard filter  (curriculum=False): keep only top-quality_frac synthetic
    samples for all epochs.
  • Curriculum   (curriculum=True):  start with top-quality_frac synthetic
    samples, linearly expand to 100 % by the last epoch so the model first
    sees the best synthetic examples before harder ones are introduced.
"""

import os
import numpy as np
import pandas as pd
import torch
from random import shuffle
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.data import Data

from util.bold import get_fc

DEFAULT_TS_FILENAME_SUFFIX = "_shen268_ts.npy"


def _load_roi_ts_from_pt(path: str, transpose_if_first_smaller_than_second: bool = True) -> np.ndarray:
    """Load DK atlas ROI time series from a `.pt` file and return float32 (T, V)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        x = obj.detach().cpu().float().numpy()
    elif isinstance(obj, np.ndarray):
        x = obj.astype(np.float32, copy=False)
    elif isinstance(obj, dict):
        for key in ("roi_ts", "ts", "time_series", "data", "tensor", "values"):
            if key in obj:
                val = obj[key]
                if isinstance(val, torch.Tensor):
                    x = val.detach().cpu().float().numpy()
                else:
                    x = np.asarray(val, dtype=np.float32)
                break
        else:
            raise ValueError(f"Unsupported .pt dict contents at {path!r}: keys={list(obj.keys())[:10]}")
    else:
        x = np.asarray(obj, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D ROI ts in {path!r}; got shape={x.shape}")
    if transpose_if_first_smaller_than_second and x.shape[0] < x.shape[1]:
        x = x.T
    return x.astype(np.float32, copy=False)


class DatasetBiopointRest(Dataset):
    """
    Biopoint resting-state ROI time series for autism classification.
    __getitem__ returns a PyG Data with x, edge_index, edge_attr, y built from dynamic FC.
    """

    def __init__(
        self,
        sourcedir,
        csv_path=None,
        k_fold=5,
        train_ratio=0.8,
        window_size=50,
        window_stride=3,
        window_num=12,
        dynamic_length=None,
        ts_filename_suffix=None,
        use_rest=True,
        top_k_edges=50,
        atlas_source: str = "shen268",
        dk_atlas_ts_root: str | None = None,
    ):
        super().__init__()
        self.sourcedir = os.path.abspath(sourcedir)
        self.ts_filename_suffix = ts_filename_suffix or DEFAULT_TS_FILENAME_SUFFIX
        self.dynamic_length = dynamic_length
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_num = window_num
        self.use_rest = use_rest
        self.k_fold = k_fold
        self.train_ratio = train_ratio
        self.top_k_edges = top_k_edges  # keep top-K strongest |FC| edges per node (per snapshot)
        self.atlas_source = atlas_source
        self.dk_atlas_ts_root = os.path.abspath(dk_atlas_ts_root) if dk_atlas_ts_root else "./data/biopoint_dk_atlas"
        self.k = None

        self.data_root = self.sourcedir
        self.output_dir = os.path.join(self.data_root, "output")
        if csv_path is None:
            csv_path = os.path.join(self.data_root, "biopoint_data.csv")
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(self.data_root, csv_path)
        self.patient_df = pd.read_csv(csv_path)

        self.full_subject_list = []
        self.full_label_list = []
        for idx in range(len(self.patient_df)):
            sub_id = str(self.patient_df.iloc[idx]["subject_id"])
            subdir = "rest" if use_rest else "task"
            if self.atlas_source == "dk":
                path = os.path.join(self.dk_atlas_ts_root, f"{sub_id}_{subdir}_roi_ts.pt")
            else:
                suffix = self.ts_filename_suffix
                fname = f"{sub_id}{suffix}" if suffix.startswith("_") else f"{sub_id}_{suffix}"
                path = os.path.join(self.output_dir, sub_id, subdir, fname)
            if not os.path.exists(path):
                continue
            try:
                ts = _load_roi_ts_from_pt(path) if self.atlas_source == "dk" else np.load(path)
                if ts.ndim != 2 or ts.shape[1] < 2:
                    continue
            except Exception:
                continue
            group = self.patient_df.iloc[idx]["group"]
            label = 1 if str(group).strip().lower() in ("pat", "patient", "asd", "1") else 0
            self.full_subject_list.append(sub_id)
            self.full_label_list.append(label)

        if not self.full_subject_list:
            raise FileNotFoundError(
                f"No valid ROI time series found under {self.output_dir} with suffix {self.ts_filename_suffix}."
            )
        self.patient_df = self.patient_df[
            self.patient_df["subject_id"].astype(str).isin(self.full_subject_list)
        ].reset_index(drop=True)
        self.behavioral_dict = dict(zip(self.full_subject_list, self.full_label_list))

        # Infer num_nodes from first subject
        first_sub = self.full_subject_list[0]
        subdir0 = "rest" if use_rest else "task"
        if self.atlas_source == "dk":
            first_ts_path = os.path.join(self.dk_atlas_ts_root, f"{first_sub}_{subdir0}_roi_ts.pt")
            first_ts = _load_roi_ts_from_pt(first_ts_path)
        else:
            suffix = self.ts_filename_suffix
            fname0 = f"{first_sub}{suffix}" if suffix.startswith("_") else f"{first_sub}_{suffix}"
            first_ts = np.load(os.path.join(self.output_dir, first_sub, subdir0, fname0))
        self.num_timepoints, self.num_nodes = first_ts.shape
        self.num_classes = 2

        if k_fold > 1:
            self.folds = list(range(k_fold))
            self._skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=0)
            self._train_subjects = self._test_subjects = None
        else:
            self.folds = [0]
            self._skf = None
            tr, te = train_test_split(
                np.arange(len(self.full_subject_list)),
                test_size=1.0 - train_ratio,
                stratify=self.full_label_list,
                random_state=0,
            )
            self._train_subjects = [self.full_subject_list[i] for i in tr]
            self._test_subjects = [self.full_subject_list[i] for i in te]
            print(f"  Train/test split: {len(self._train_subjects)} train, {len(self._test_subjects)} test (ratio={train_ratio})")
        self.subject_list = self.full_subject_list
        self.train = True
        self._quality_active_list = None   # set by set_quality_curriculum()

        n_total = len(self.full_subject_list)
        print(
            f"GCN/GAT Biopoint dataset: {n_total} subjects, "
            f"{self.num_nodes} ROIs, {self.num_timepoints} timepoints, "
            f"window_num={window_num}, k_fold={k_fold}, top_k_edges={top_k_edges}"
        )

    def __len__(self):
        active = getattr(self, "_quality_active_list", None)
        return len(active) if active is not None else len(self.subject_list)

    @property
    def _active_subjects(self):
        active = getattr(self, "_quality_active_list", None)
        return active if active is not None else self.subject_list

    def set_fold(self, fold, train=True):
        self.k = fold
        self.train = train
        if self._skf is None:
            self.subject_list = self._train_subjects if train else self._test_subjects
            return
        train_idx, test_idx = list(self._skf.split(self.full_subject_list, self.full_label_list))[fold]
        if train:
            shuffle(train_idx)
            self.subject_list = [self.full_subject_list[i] for i in train_idx]
        else:
            self.subject_list = [self.full_subject_list[i] for i in test_idx]
        # Reset any quality filter so it is recomputed for the new fold's subject list
        self._quality_active_list = None

    # ------------------------------------------------------------------
    # Quality scoring & curriculum
    # ------------------------------------------------------------------

    def _fc_quality_score(self, subject_id: str) -> float:
        """
        Score a subject by mean absolute off-diagonal FC of their full time series.
        Higher = stronger, more reliable connectivity signal.
        """
        ts = self._load_timeseries(subject_id)          # (T, V), already z-scored
        ts_t = torch.tensor(ts, dtype=torch.float32)    # (T, V)
        # Full-scan FC: (V, V)
        fc = torch.corrcoef(ts_t.T)                     # (V, V)
        mask = ~torch.eye(fc.shape[0], dtype=torch.bool)
        return fc.abs()[mask].mean().item()

    def compute_quality_scores(self, subject_ids=None):
        """
        Compute and cache quality scores for a list of subject IDs.
        Returns a dict {subject_id: score}.
        """
        if subject_ids is None:
            subject_ids = self.subject_list
        scores = {}
        for sid in subject_ids:
            try:
                scores[sid] = self._fc_quality_score(sid)
            except Exception:
                scores[sid] = 0.0
        return scores

    def set_quality_curriculum(self, quality_frac: float = 1.0):
        """
        Filter subject_list to the top-quality_frac fraction ranked by FC quality.
        Call once per epoch (or once before training for hard filtering).
        quality_frac=1.0  → use all subjects (no filtering).
        quality_frac=0.5  → keep only the top 50 % highest-quality subjects.
        Class balance is preserved: top-quality_frac applied within each class.
        """
        if quality_frac >= 1.0:
            self._quality_active_list = None   # use full subject_list
            return

        base_list = self.subject_list
        scores = self.compute_quality_scores(base_list)

        # Stratify: apply cutoff within each class to preserve balance
        class0 = [s for s in base_list if self.behavioral_dict[s] == 0]
        class1 = [s for s in base_list if self.behavioral_dict[s] == 1]

        def top_frac(subjects, frac):
            ranked = sorted(subjects, key=lambda s: scores[s], reverse=True)
            n_keep = max(1, int(len(ranked) * frac))
            return ranked[:n_keep]

        kept = top_frac(class0, quality_frac) + top_frac(class1, quality_frac)
        self._quality_active_list = kept
        print(
            f"  [quality] frac={quality_frac:.2f} → {len(kept)}/{len(base_list)} subjects kept "
            f"(class0={len(top_frac(class0, quality_frac))}, class1={len(top_frac(class1, quality_frac))})"
        )

    def _load_timeseries(self, subject_id):
        subdir = "rest" if self.use_rest else "task"
        if self.atlas_source == "dk":
            path = os.path.join(self.dk_atlas_ts_root, f"{subject_id}_{subdir}_roi_ts.pt")
            ts = _load_roi_ts_from_pt(path).astype(np.float32)
        else:
            suffix = self.ts_filename_suffix
            fname = f"{subject_id}{suffix}" if suffix.startswith("_") else f"{subject_id}_{suffix}"
            path = os.path.join(self.output_dir, subject_id, subdir, fname)
            ts = np.load(path).astype(np.float32)
        if ts.ndim == 1:
            ts = ts.reshape(-1, 1)
        ts = (ts - np.mean(ts, axis=0, keepdims=True)) / (np.std(ts, axis=0, keepdims=True) + 1e-9)
        return ts

    def _fc_to_graph(self, fc: torch.Tensor, snapshot_offset: int, roi_num: int):
        """
        Build a sparse edge_index and edge_attr for one FC snapshot.
        fc: (roi_num, roi_num) with diagonal removed.

        To avoid OOM with fully-connected graphs (268*267 ≈ 71k edges per snapshot),
        we keep only the top-K strongest |FC| neighbours per node.
        With top_k_edges=50 this gives ≤ 268*50 = 13.4k directed edges per snapshot
        (vs 71k for fully-connected), a ~5× reduction.
        """
        k = min(self.top_k_edges, roi_num - 1)
        abs_fc = fc.abs()
        # top-k neighbours for each row (node), ignoring self
        topk_vals, topk_idx = torch.topk(abs_fc, k, dim=1)  # (roi_num, k)

        rows = torch.arange(roi_num).unsqueeze(1).expand_as(topk_idx)  # (roi_num, k)
        src = (rows.reshape(-1) + snapshot_offset).long()
        dst = (topk_idx.reshape(-1) + snapshot_offset).long()
        attr = fc[rows.reshape(-1), topk_idx.reshape(-1)].float().unsqueeze(1)

        edge_index = torch.stack([src, dst], dim=0)
        return edge_index, attr

    def _timeseries_to_data(self, timeseries: torch.Tensor, label: int) -> Data:
        T, roi_num = timeseries.shape
        dynamic_length = self.dynamic_length or T
        if dynamic_length is not None and T >= dynamic_length and getattr(self, "train", True):
            from random import randrange
            start = randrange(T - dynamic_length + 1)
            timeseries = timeseries[start: start + dynamic_length]
            T = dynamic_length
        if T < self.window_size:
            timeseries = torch.nn.functional.pad(timeseries, (0, 0, 0, self.window_size - T))
            T = timeseries.shape[0]
        max_start = max(0, T - self.window_size)
        candidates = list(range(0, max_start + 1, self.window_stride))
        if not candidates:
            candidates = [0]
        sampling_points = [candidates[min(i, len(candidates) - 1)] for i in range(self.window_num)]

        x_list, edge_index_list, edge_attr_list = [], [], []
        for s, start in enumerate(sampling_points):
            fc = get_fc(timeseries, start, self.window_size, self_loop=False)
            x_list.append(fc)
            ei, ea = self._fc_to_graph(fc, s * roi_num, roi_num)
            edge_index_list.append(ei)
            edge_attr_list.append(ea)

        return Data(
            x=torch.cat(x_list, dim=0),
            edge_index=torch.cat(edge_index_list, dim=1),
            edge_attr=torch.cat(edge_attr_list, dim=0),
            y=torch.tensor([label], dtype=torch.long),
        )

    def __getitem__(self, idx):
        subject = self._active_subjects[idx]
        ts = self._load_timeseries(subject)
        ts_tensor = torch.tensor(ts, dtype=torch.float32)
        label = self.behavioral_dict[subject]
        return self._timeseries_to_data(ts_tensor, label)


class DatasetBiopointRestWithSynthetic(DatasetBiopointRest):
    """Real + synthetic training set; test set is always real-only."""

    def __init__(
        self,
        sourcedir,
        synthetic_dir,
        csv_path=None,
        k_fold=5,
        train_ratio=0.8,
        window_size=50,
        window_stride=3,
        window_num=12,
        dynamic_length=None,
        ts_filename_suffix=None,
        use_rest=True,
        synthetic_manifest_path=None,
        top_k_edges=50,
        atlas_source: str = "shen268",
        dk_atlas_ts_root: str | None = None,
    ):
        super().__init__(
            sourcedir=sourcedir,
            csv_path=csv_path,
            k_fold=k_fold,
            train_ratio=train_ratio,
            window_size=window_size,
            window_stride=window_stride,
            window_num=window_num,
            dynamic_length=dynamic_length,
            ts_filename_suffix=ts_filename_suffix,
            use_rest=use_rest,
            top_k_edges=top_k_edges,
            atlas_source=atlas_source,
            dk_atlas_ts_root=dk_atlas_ts_root,
        )
        self.synthetic_dir = os.path.abspath(synthetic_dir)
        manifest_path = synthetic_manifest_path or os.path.join(self.synthetic_dir, "synthetic_manifest.csv")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Synthetic manifest not found: {manifest_path}")
        syn_df = pd.read_csv(manifest_path)
        self.synthetic_list = []
        n_missing = 0
        for _, row in syn_df.iterrows():
            sub_id = str(row["subject_id"]).strip()
            path = row.get("path")
            if pd.isna(path) or str(path).strip() == "":
                # No path in manifest — use default naming
                path = os.path.join(self.synthetic_dir, f"{sub_id}_syn.npy")
            else:
                path = str(path).strip()
                # If the manifest path is relative, try resolving it against
                # synthetic_dir (handles the case where the manifest was written
                # with a relative path from a different working directory)
                if not os.path.isabs(path):
                    resolved = os.path.join(self.synthetic_dir, os.path.basename(path))
                    if os.path.isfile(resolved):
                        path = resolved
                    # else: keep original relative path and let the isfile check below handle it
            if not os.path.isfile(path):
                n_missing += 1
                continue
            label = int(row.get("label", 0))
            self.synthetic_list.append((sub_id, path, label))
        if n_missing > 0:
            print(f"  WARNING: {n_missing}/{len(syn_df)} synthetic files not found on disk (skipped).")
        if self.synthetic_list:
            first_syn = np.load(self.synthetic_list[0][1])
            if first_syn.shape[1] != self.num_nodes:
                raise ValueError(
                    f"Synthetic num_roi={first_syn.shape[1]} != real num_nodes={self.num_nodes}."
                )
        self._train_real_ids = None
        self._train_synthetic_items = None
        self._test_real_ids = None
        self._active_real_ids = None        # unused — real data always kept in full
        self._active_synthetic_items = None  # set by set_quality_curriculum(); None = use all synthetic

        n_syn_total = len(self.synthetic_list)
        if self._train_subjects is not None:
            train_set = set(self._train_subjects)
            n_syn_used = sum(1 for (sid, _, _) in self.synthetic_list if sid in train_set)
            print(
                f"  + {n_syn_used} synthetic for training "
                f"(from {n_syn_total} loaded; test-subject synthetics excluded)"
            )
        else:
            print(f"  + {n_syn_total} synthetic in manifest (filtered to training subjects per fold)")

    def set_fold(self, fold, train=True):
        self.k = fold
        self.train = train
        self._active_real_ids = None       # unused
        self._active_synthetic_items = None  # reset quality filter on fold change
        if self._skf is None:
            if train:
                self._train_real_ids = self._train_subjects
                train_ids_set = set(self._train_real_ids)
                self._train_synthetic_items = [
                    (sid, path, lab) for (sid, path, lab) in self.synthetic_list if sid in train_ids_set
                ]
                self._test_real_ids = None
                self.subject_list = self._train_real_ids
            else:
                self._test_real_ids = self._test_subjects
                self._train_real_ids = None
                self._train_synthetic_items = None
                self.subject_list = self._test_real_ids
            return
        train_idx, test_idx = list(self._skf.split(self.full_subject_list, self.full_label_list))[fold]
        if train:
            shuffle(train_idx)
            self._train_real_ids = [self.full_subject_list[i] for i in train_idx]
            train_ids_set = set(self._train_real_ids)
            self._train_synthetic_items = [
                (sid, path, lab) for (sid, path, lab) in self.synthetic_list if sid in train_ids_set
            ]
            self._test_real_ids = None
            self.subject_list = self._train_real_ids
        else:
            self._test_real_ids = [self.full_subject_list[i] for i in test_idx]
            self._train_real_ids = None
            self._train_synthetic_items = None
            self.subject_list = self._test_real_ids

    def _fc_quality_score_from_path(self, path: str) -> float:
        """Score a synthetic .npy file by mean absolute off-diagonal FC."""
        ts = np.load(path).astype(np.float32)
        if ts.ndim == 1:
            ts = ts.reshape(-1, 1)
        ts = (ts - np.mean(ts, axis=0, keepdims=True)) / (np.std(ts, axis=0, keepdims=True) + 1e-9)
        ts_t = torch.tensor(ts, dtype=torch.float32)
        fc = torch.corrcoef(ts_t.T)
        mask = ~torch.eye(fc.shape[0], dtype=torch.bool)
        return fc.abs()[mask].mean().item()

    def set_quality_curriculum(self, quality_frac: float = 1.0):
        """
        Override: quality filtering applies ONLY to synthetic training items.
        Real subjects are always kept in full.
        Synthetic samples are ranked by FC signal quality and only the top
        quality_frac fraction (per class) are used each epoch.
        """
        if not self.train or self._train_synthetic_items is None:
            self._active_real_ids = None
            return

        # Real data: always use all of them
        self._active_real_ids = None  # None = use full _train_real_ids

        if quality_frac >= 1.0:
            self._active_synthetic_items = None  # use all synthetic
            return

        # Score each synthetic sample
        scores = {}
        for (sid, path, label) in self._train_synthetic_items:
            try:
                scores[path] = self._fc_quality_score_from_path(path)
            except Exception:
                scores[path] = 0.0

        # Filter within each class to preserve balance
        class0 = [(sid, path, lab) for (sid, path, lab) in self._train_synthetic_items if lab == 0]
        class1 = [(sid, path, lab) for (sid, path, lab) in self._train_synthetic_items if lab == 1]

        def top_frac(items, frac):
            ranked = sorted(items, key=lambda x: scores[x[1]], reverse=True)
            n_keep = max(1, int(len(ranked) * frac))
            return ranked[:n_keep]

        kept_syn = top_frac(class0, quality_frac) + top_frac(class1, quality_frac)
        self._active_synthetic_items = kept_syn
        print(
            f"  [quality] synthetic frac={quality_frac:.2f} → "
            f"{len(kept_syn)}/{len(self._train_synthetic_items)} synthetic kept "
            f"(class0={len(top_frac(class0, quality_frac))}, class1={len(top_frac(class1, quality_frac))}) "
            f"+ {len(self._train_real_ids)} real (unfiltered)"
        )

    @property
    def _effective_synthetic_items(self):
        """Synthetic items actually used this epoch (quality-filtered or full)."""
        active = getattr(self, "_active_synthetic_items", None)
        if active is not None:
            return active
        return self._train_synthetic_items if self._train_synthetic_items is not None else []

    def __len__(self):
        if not self.train or self._train_real_ids is None:
            return len(self.subject_list)
        # All real + quality-filtered synthetic
        return len(self._train_real_ids) + len(self._effective_synthetic_items)

    def __getitem__(self, idx):
        if self.train and self._train_real_ids is not None and self._train_synthetic_items is not None:
            n_real = len(self._train_real_ids)
            if idx < n_real:
                subject = self._train_real_ids[idx]
                ts = self._load_timeseries(subject)
                ts_tensor = torch.tensor(ts, dtype=torch.float32)
                label = self.behavioral_dict[subject]
                return self._timeseries_to_data(ts_tensor, label)
            else:
                syn_idx = idx - n_real
                _sid, path, label = self._effective_synthetic_items[syn_idx]
                ts = np.load(path).astype(np.float32)
                if ts.ndim == 1:
                    ts = ts.reshape(-1, 1)
                ts = (ts - np.mean(ts, axis=0, keepdims=True)) / (np.std(ts, axis=0, keepdims=True) + 1e-9)
                if self.dynamic_length is not None:
                    if len(ts) >= self.dynamic_length:
                        from random import randrange
                        start = randrange(len(ts) - self.dynamic_length + 1)
                        ts = ts[start: start + self.dynamic_length]
                    else:
                        pad = np.zeros((self.dynamic_length - len(ts), ts.shape[1]), dtype=np.float32)
                        ts = np.vstack([ts, pad])
                ts_tensor = torch.tensor(ts, dtype=torch.float32)
                return self._timeseries_to_data(ts_tensor, label)
        # Test mode — real only
        subject = self.subject_list[idx]
        ts = self._load_timeseries(subject)
        ts_tensor = torch.tensor(ts, dtype=torch.float32)
        label = self.behavioral_dict[subject]
        return self._timeseries_to_data(ts_tensor, label)
