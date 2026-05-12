"""
Biopoint dataset for STAGIN: autism classification using AAL (or other) ROI time series.
Expects pre-extracted ROI time series, e.g. data_root/output/{subject_id}/rest/{subject_id}_aal_ts.npy
with shape (T, num_roi). Compatible with STAGIN's train/test interface (set_fold, __getitem__ with
'timeseries', 'label', 'id').
"""

import os
import numpy as np
import pandas as pd
import torch
from random import shuffle
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold, train_test_split


# Biopoint fm-fmri uses shen268; use ts_filename_suffix for AAL or other atlases
DEFAULT_TS_FILENAME_SUFFIX = "_shen268_ts.npy"  # file: {subject_id}_shen268_ts.npy under data_root/output/<id>/rest/


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
    Biopoint resting-state ROI time series for autism classification (pat vs control).
    Uses pre-extracted ROI time series (e.g. AAL). Same interface as STAGIN's DatasetHCPRest:
    set_fold(k, train=True/False), __getitem__ -> {'id', 'timeseries', 'label'}, num_nodes, num_classes, folds.
    """

    def __init__(
        self,
        sourcedir,
        csv_path=None,
        k_fold=5,
        train_ratio=0.8,
        dynamic_length=None,
        ts_filename_suffix=None,
        use_rest=True,
        atlas_source: str = "shen268",
        dk_atlas_ts_root: str | None = None,
    ):
        super().__init__()
        self.sourcedir = os.path.abspath(sourcedir)
        self.ts_filename_suffix = ts_filename_suffix or DEFAULT_TS_FILENAME_SUFFIX
        self.dynamic_length = dynamic_length
        self.use_rest = use_rest
        self.k_fold = k_fold
        self.train_ratio = train_ratio
        self.atlas_source = atlas_source
        self.dk_atlas_ts_root = os.path.abspath(dk_atlas_ts_root) if dk_atlas_ts_root else "./data/biopoint_dk_atlas"
        self.k = None

        # data_root from STAGIN-style sourcedir: we treat sourcedir as data root (e.g. biopoint_data)
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
                fname = f"{sub_id}{self.ts_filename_suffix}" if self.ts_filename_suffix.startswith("_") else f"{sub_id}_{self.ts_filename_suffix}"
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
                f"No valid ROI time series found under {self.output_dir} with suffix {self.ts_filename_suffix}. "
                "Check sourcedir and ts_filename_suffix (e.g. _aal_ts.npy)."
            )
        self.patient_df = self.patient_df[self.patient_df["subject_id"].astype(str).isin(self.full_subject_list)].reset_index(drop=True)
        self.behavioral_dict = dict(zip(self.full_subject_list, self.full_label_list))

        # Infer num_nodes and num_timepoints from first subject
        subdir = "rest" if use_rest else "task"
        s0 = self.full_subject_list[0]
        if self.atlas_source == "dk":
            first_ts_path = os.path.join(self.dk_atlas_ts_root, f"{s0}_{subdir}_roi_ts.pt")
            first_ts = _load_roi_ts_from_pt(first_ts_path)
        else:
            fname0 = f"{s0}{self.ts_filename_suffix}" if self.ts_filename_suffix.startswith("_") else f"{s0}_{self.ts_filename_suffix}"
            first_ts_path = os.path.join(self.output_dir, s0, subdir, fname0)
            first_ts = np.load(first_ts_path)
        self.num_timepoints, self.num_nodes = first_ts.shape
        self.num_classes = 2  # autism vs control

        if k_fold > 1:
            self.folds = list(range(k_fold))
            self._skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=0)
            self._train_subjects = self._test_subjects = None
        else:
            self.folds = [0]
            self._skf = None
            # Single stratified train/test split
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
        self.train = True  # for dynamic_length random crop in __getitem__

        n_total = len(self.full_subject_list)
        print(
            f"Biopoint STAGIN dataset: {n_total} subjects total, "
            f"{self.num_nodes} ROIs, {self.num_timepoints} timepoints, "
            f"classes={self.num_classes}, k_fold={k_fold}"
        )

    def __len__(self):
        return len(self.subject_list)

    def set_fold(self, fold, train=True):
        self.k = fold
        self.train = train
        if self._skf is None:
            # k_fold=1: use precomputed train/test split
            self.subject_list = self._train_subjects if train else self._test_subjects
            return
        train_idx, test_idx = list(self._skf.split(self.full_subject_list, self.full_label_list))[fold]
        if train:
            shuffle(train_idx)
            self.subject_list = [self.full_subject_list[i] for i in train_idx]
        else:
            self.subject_list = [self.full_subject_list[i] for i in test_idx]

    def _load_timeseries(self, subject_id):
        subdir = "rest" if self.use_rest else "task"
        if self.atlas_source == "dk":
            path = os.path.join(self.dk_atlas_ts_root, f"{subject_id}_{subdir}_roi_ts.pt")
            ts = _load_roi_ts_from_pt(path).astype(np.float32)
        else:
            fname = f"{subject_id}{self.ts_filename_suffix}" if self.ts_filename_suffix.startswith("_") else f"{subject_id}_{self.ts_filename_suffix}"
            path = os.path.join(self.output_dir, subject_id, subdir, fname)
            ts = np.load(path).astype(np.float32)
        if ts.ndim == 1:
            ts = ts.reshape(-1, 1)
        ts = (ts - np.mean(ts, axis=0, keepdims=True)) / (np.std(ts, axis=0, keepdims=True) + 1e-9)
        return ts

    def __getitem__(self, idx):
        subject = self.subject_list[idx]
        timeseries = self._load_timeseries(subject)
        if self.dynamic_length is not None and len(timeseries) >= self.dynamic_length and getattr(self, "train", True):
            from random import randrange
            start = randrange(len(timeseries) - self.dynamic_length + 1)
            timeseries = timeseries[start : start + self.dynamic_length]
        label = self.behavioral_dict[subject]
        return {
            "id": subject,
            "timeseries": torch.tensor(timeseries, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.int64),
        }


class DatasetBiopointRestWithSynthetic(DatasetBiopointRest):
    """
    Same as DatasetBiopointRest but when train=True adds synthetic samples (from fm-fmri)
    to the training set. Test set is real-only. Folds are defined on real subjects only;
    synthetic samples are appended to the training fold only.
    """

    def __init__(
        self,
        sourcedir,
        synthetic_dir,
        csv_path=None,
        k_fold=5,
        train_ratio=0.8,
        dynamic_length=None,
        ts_filename_suffix=None,
        use_rest=True,
        synthetic_manifest_path=None,
        atlas_source: str = "shen268",
        dk_atlas_ts_root: str | None = None,
    ):
        super().__init__(
            sourcedir=sourcedir,
            csv_path=csv_path,
            k_fold=k_fold,
            train_ratio=train_ratio,
            dynamic_length=dynamic_length,
            ts_filename_suffix=ts_filename_suffix,
            use_rest=use_rest,
            atlas_source=atlas_source,
            dk_atlas_ts_root=dk_atlas_ts_root,
        )
        self.synthetic_dir = os.path.abspath(synthetic_dir)
        manifest_path = synthetic_manifest_path or os.path.join(self.synthetic_dir, "synthetic_manifest.csv")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Synthetic manifest not found: {manifest_path}. Run generate_synthetic_biopoint.py first.")
        syn_df = pd.read_csv(manifest_path)
        self.synthetic_list = []  # (subject_id, path, label) for filtering by fold
        for _, row in syn_df.iterrows():
            sub_id = str(row["subject_id"])
            path = row.get("path")
            if pd.isna(path):
                path = os.path.join(self.synthetic_dir, f"{sub_id}_syn.npy")
            if not os.path.isfile(path):
                continue
            label = int(row.get("label", 0))
            self.synthetic_list.append((sub_id, path, label))
        if self.synthetic_list:
            first_syn = np.load(self.synthetic_list[0][1])
            if first_syn.shape[1] != self.num_nodes:
                raise ValueError(
                    f"Synthetic num_roi={first_syn.shape[1]} != real num_nodes={self.num_nodes}. "
                    "Use same atlas (ts_filename_suffix) for real and fm-fmri."
                )
            self.synthetic_T = first_syn.shape[0]
        else:
            self.synthetic_T = 0
        self._train_real_ids = None
        self._train_synthetic_items = None
        self._test_real_ids = None
        self._combined_train_length = 0
        self._active_synthetic_items = None  # set by set_quality_curriculum(); None = use all

        # Report how many synthetic will actually be used (only those for training subjects)
        n_syn_total = len(self.synthetic_list)
        if self._train_subjects is not None:
            train_set = set(self._train_subjects)
            n_syn_used = sum(1 for (sid, _, _) in self.synthetic_list if sid in train_set)
            print(
                f"  + {n_syn_used} synthetic samples used for training (from {n_syn_total} in manifest; test-subject synthetics excluded); synthetic T={self.synthetic_T}"
            )
        else:
            print(
                f"  + {n_syn_total} synthetic in manifest (filtered to training subjects per fold); synthetic T={self.synthetic_T}"
            )

    def set_fold(self, fold, train=True):
        self.k = fold
        self.train = train
        if self._skf is None:
            # k_fold=1: use stratified train/test split from base class
            if train:
                self._train_real_ids = self._train_subjects
                train_ids_set = set(self._train_real_ids)
                self._train_synthetic_items = [(sid, path, lab) for (sid, path, lab) in self.synthetic_list if sid in train_ids_set]
                self._test_real_ids = None
                self.subject_list = self._train_real_ids
                self._combined_train_length = len(self._train_real_ids) + len(self._train_synthetic_items)
            else:
                self._test_real_ids = self._test_subjects
                self._train_real_ids = None
                self._train_synthetic_items = None
                self.subject_list = self._test_real_ids
                self._combined_train_length = 0
            return
        train_idx, test_idx = list(self._skf.split(self.full_subject_list, self.full_label_list))[fold]
        if train:
            shuffle(train_idx)
            self._train_real_ids = [self.full_subject_list[i] for i in train_idx]
            train_ids_set = set(self._train_real_ids)
            # Only use synthetic from training subjects to avoid leaking test-subject information
            self._train_synthetic_items = [(sid, path, lab) for (sid, path, lab) in self.synthetic_list if sid in train_ids_set]
            self._test_real_ids = None
            self.subject_list = self._train_real_ids
            self._combined_train_length = len(self._train_real_ids) + len(self._train_synthetic_items)
        else:
            self._test_real_ids = [self.full_subject_list[i] for i in test_idx]
            self._train_real_ids = None
            self._train_synthetic_items = None
            self.subject_list = self._test_real_ids
            self._combined_train_length = 0

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
        Filter synthetic training items to the top quality_frac fraction
        ranked by FC signal quality. Real subjects are always kept in full.
        """
        if not self.train or self._train_synthetic_items is None:
            self._active_synthetic_items = None
            return

        if quality_frac >= 1.0:
            self._active_synthetic_items = None  # use all synthetic
            self._combined_train_length = len(self._train_real_ids) + len(self._train_synthetic_items)
            return

        scores = {}
        for (sid, path, label) in self._train_synthetic_items:
            try:
                scores[path] = self._fc_quality_score_from_path(path)
            except Exception:
                scores[path] = 0.0

        class0 = [(sid, path, lab) for (sid, path, lab) in self._train_synthetic_items if lab == 0]
        class1 = [(sid, path, lab) for (sid, path, lab) in self._train_synthetic_items if lab == 1]

        def top_frac(items, frac):
            ranked = sorted(items, key=lambda x: scores[x[1]], reverse=True)
            n_keep = max(1, int(len(ranked) * frac))
            return ranked[:n_keep]

        kept_syn = top_frac(class0, quality_frac) + top_frac(class1, quality_frac)
        self._active_synthetic_items = kept_syn
        self._combined_train_length = len(self._train_real_ids) + len(kept_syn)
        print(
            f"  [quality] synthetic frac={quality_frac:.2f} → "
            f"{len(kept_syn)}/{len(self._train_synthetic_items)} synthetic kept "
            f"+ {len(self._train_real_ids)} real (unfiltered)"
        )

    @property
    def _effective_synthetic_items(self):
        """Synthetic items actually used this epoch (quality-filtered or full)."""
        if self._active_synthetic_items is not None:
            return self._active_synthetic_items
        return self._train_synthetic_items if self._train_synthetic_items is not None else []

    def __len__(self):
        if self._skf is None and not (self.train and self._train_real_ids is not None):
            return len(self.subject_list)
        if self.train and self._train_real_ids is not None:
            return len(self._train_real_ids) + len(self._effective_synthetic_items)
        return len(self.subject_list)

    def __getitem__(self, idx):
        if self.train and self._train_real_ids is not None and self._train_synthetic_items is not None:
            if idx < len(self._train_real_ids):
                subject = self._train_real_ids[idx]
                timeseries = self._load_timeseries(subject)
                if self.dynamic_length is not None and len(timeseries) >= self.dynamic_length:
                    from random import randrange
                    start = randrange(len(timeseries) - self.dynamic_length + 1)
                    timeseries = timeseries[start : start + self.dynamic_length]
                label = self.behavioral_dict[subject]
                return {
                    "id": subject,
                    "timeseries": torch.tensor(timeseries, dtype=torch.float32),
                    "label": torch.tensor(label, dtype=torch.int64),
                }
            else:
                syn_idx = idx - len(self._train_real_ids)
                _sid, path, label = self._effective_synthetic_items[syn_idx]
                ts = np.load(path).astype(np.float32)
                if ts.ndim == 1:
                    ts = ts.reshape(-1, 1)
                ts = (ts - np.mean(ts, axis=0, keepdims=True)) / (np.std(ts, axis=0, keepdims=True) + 1e-9)
                target_len = self.dynamic_length or self.num_timepoints
                if len(ts) >= target_len:
                    from random import randrange
                    start = randrange(len(ts) - target_len + 1)
                    ts = ts[start : start + target_len]
                else:
                    pad = np.zeros((target_len - len(ts), ts.shape[1]), dtype=np.float32)
                    ts = np.vstack([ts, pad])
                return {
                    "id": f"syn_{syn_idx}",
                    "timeseries": torch.tensor(ts, dtype=torch.float32),
                    "label": torch.tensor(label, dtype=torch.int64),
                }
        # Test or no synthetic
        subject = self.subject_list[idx]
        timeseries = self._load_timeseries(subject)
        if self.dynamic_length is not None and len(timeseries) >= self.dynamic_length and getattr(self, "train", True):
            from random import randrange
            start = randrange(len(timeseries) - self.dynamic_length + 1)
            timeseries = timeseries[start : start + self.dynamic_length]
        label = self.behavioral_dict[subject]
        return {
            "id": subject,
            "timeseries": torch.tensor(timeseries, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.int64),
        }
