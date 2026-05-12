"""
Biopoint dataset for GAT: load ROI time series and convert to PyG Data (dynamic FC -> graph per snapshot).
Supports real-only or real + synthetic for data augmentation.
Bootstrap augmentation: each real subject is replicated bootstrap_n times with randomly
resampled temporal windows (with replacement), following the STNAGNN paper's strategy.
"""

import os
import random as _random
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
        atlas_source: str = "shen268",
        dk_atlas_ts_root: str | None = None,
        sparsity: int = 0,
        noise_level: float = 0.0,
        bootstrap_n: int = 0,
    ):
        super().__init__()
        self.sparsity = sparsity
        self.noise_level = noise_level
        self.bootstrap_n = bootstrap_n
        self.sourcedir = os.path.abspath(sourcedir)
        self.ts_filename_suffix = ts_filename_suffix or DEFAULT_TS_FILENAME_SUFFIX
        self.dynamic_length = dynamic_length
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_num = window_num
        self.use_rest = use_rest
        self.atlas_source = atlas_source
        self.dk_atlas_ts_root = os.path.abspath(dk_atlas_ts_root) if dk_atlas_ts_root else "./data/biopoint_dk_atlas"
        self.k_fold = k_fold
        self.train_ratio = train_ratio
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
            if self.atlas_source == "dk":
                subdir = "rest" if use_rest else "task"
                path = os.path.join(self.dk_atlas_ts_root, f"{sub_id}_{subdir}_roi_ts.pt")
            else:
                subdir = "rest" if use_rest else "task"
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
                f"No valid ROI time series found under {self.output_dir} with suffix {self.ts_filename_suffix}."
            )
        self.patient_df = self.patient_df[self.patient_df["subject_id"].astype(str).isin(self.full_subject_list)].reset_index(drop=True)
        self.behavioral_dict = dict(zip(self.full_subject_list, self.full_label_list))

        if self.atlas_source == "dk":
            first_subdir = "rest" if use_rest else "task"
            first_ts_path = os.path.join(self.dk_atlas_ts_root, f"{self.full_subject_list[0]}_{first_subdir}_roi_ts.pt")
            first_ts = _load_roi_ts_from_pt(first_ts_path)
        else:
            first_ts_path = os.path.join(
                self.output_dir,
                self.full_subject_list[0],
                "rest" if use_rest else "task",
                f"{self.full_subject_list[0]}{self.ts_filename_suffix}" if self.ts_filename_suffix.startswith("_") else f"{self.full_subject_list[0]}_{self.ts_filename_suffix}",
            )
            first_ts = np.load(first_ts_path)
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

        n_total = len(self.full_subject_list)
        print(
            f"GAT Biopoint dataset: {n_total} subjects total, "
            f"{self.num_nodes} ROIs, {self.num_timepoints} timepoints, "
            f"window_num={window_num}, k_fold={k_fold}"
        )

    def __len__(self):
        n = len(self.subject_list)
        if self.train and self.bootstrap_n > 0:
            return n * (1 + self.bootstrap_n)
        return n

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

    @staticmethod
    def _fc_to_graph(fc: torch.Tensor, snapshot_offset: int, roi_num: int, sparsity: int = 0):
        """Build edge_index and edge_attr for one FC snapshot. fc (roi_num, roi_num).
        sparsity: percentile threshold — edges with abs(weight) below this percentile are dropped."""
        mask = ~torch.eye(roi_num, dtype=torch.bool, device=fc.device)
        flat_vals = fc[mask]

        if sparsity > 0:
            threshold = torch.quantile(flat_vals.abs().float(), sparsity / 100.0)
            keep = flat_vals.abs() >= threshold
        else:
            keep = torch.ones_like(flat_vals, dtype=torch.bool)

        rows, cols = mask.nonzero(as_tuple=True)
        rows = rows[keep] + snapshot_offset
        cols = cols[keep] + snapshot_offset
        vals = flat_vals[keep]
        edge_index = torch.stack([rows, cols], dim=0).long()
        edge_attr = vals.unsqueeze(1)
        return edge_index, edge_attr

    def _timeseries_to_data(self, timeseries: torch.Tensor, label: int, bootstrap: bool = False) -> Data:
        T, roi_num = timeseries.shape
        dynamic_length = self.dynamic_length or T
        if dynamic_length is not None and T >= dynamic_length and getattr(self, "train", True):
            start = _random.randrange(T - dynamic_length + 1)
            timeseries = timeseries[start : start + dynamic_length]
            T = dynamic_length
        if self.noise_level > 0 and getattr(self, "train", True):
            timeseries = timeseries + torch.randn_like(timeseries) * self.noise_level
        if T < self.window_size:
            timeseries = torch.nn.functional.pad(timeseries, (0, 0, 0, self.window_size - T))
            T = timeseries.shape[0]
        max_start = max(0, T - self.window_size)
        candidates = list(range(0, max_start + 1, self.window_stride))
        if not candidates:
            candidates = [0]

        if bootstrap:
            sampling_points = _random.choices(candidates, k=self.window_num)
        else:
            sampling_points = [candidates[min(i, len(candidates) - 1)] for i in range(self.window_num)]
        window_num = self.window_num

        x_list = []
        edge_index_list = []
        edge_attr_list = []
        for s, start in enumerate(sampling_points):
            fc = get_fc(timeseries, start, self.window_size, self_loop=False)
            # Node features = rows of FC (roi_num, roi_num)
            x_list.append(fc)
            offset = s * roi_num
            ei, ea = self._fc_to_graph(fc, offset, roi_num, sparsity=self.sparsity)
            edge_index_list.append(ei)
            edge_attr_list.append(ea)

        x = torch.cat(x_list, dim=0)
        edge_index = torch.cat(edge_index_list, dim=1)
        edge_attr = torch.cat(edge_attr_list, dim=0)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([label], dtype=torch.long),
        )

    def __getitem__(self, idx):
        n_base = len(self.subject_list)
        is_bootstrap = self.train and self.bootstrap_n > 0 and idx >= n_base
        actual_idx = idx % n_base if is_bootstrap else idx
        subject = self.subject_list[actual_idx]
        timeseries = self._load_timeseries(subject)
        ts_tensor = torch.tensor(timeseries, dtype=torch.float32)
        label = self.behavioral_dict[subject]
        return self._timeseries_to_data(ts_tensor, label, bootstrap=is_bootstrap)


class DatasetBiopointRestWithSynthetic(DatasetBiopointRest):
    """Real + synthetic training set; test set is real-only."""

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
        atlas_source: str = "shen268",
        dk_atlas_ts_root: str | None = None,
        sparsity: int = 0,
        noise_level: float = 0.0,
        bootstrap_n: int = 0,
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
            atlas_source=atlas_source,
            dk_atlas_ts_root=dk_atlas_ts_root,
            sparsity=sparsity,
            noise_level=noise_level,
            bootstrap_n=bootstrap_n,
        )
        self.synthetic_dir = os.path.abspath(synthetic_dir)
        manifest_path = synthetic_manifest_path or os.path.join(self.synthetic_dir, "synthetic_manifest.csv")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Synthetic manifest not found: {manifest_path}")
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
                    f"Synthetic num_roi={first_syn.shape[1]} != real num_nodes={self.num_nodes}."
                )
        self._train_real_ids = None
        self._train_synthetic_items = None
        self._test_real_ids = None
        self._combined_train_length = 0

        n_syn_total = len(self.synthetic_list)
        if self._train_subjects is not None:
            train_set = set(self._train_subjects)
            n_syn_used = sum(1 for (sid, _, _) in self.synthetic_list if sid in train_set)
            print(
                f"  + {n_syn_used} synthetic used for training (from {n_syn_total} in manifest; test-subject synthetics excluded)"
            )
        else:
            print(f"  + {n_syn_total} synthetic in manifest (filtered to training subjects per fold)")

    def set_fold(self, fold, train=True):
        self.k = fold
        self.train = train
        if self._skf is None:
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

    def __len__(self):
        if self.train and self._train_real_ids is not None and self._train_synthetic_items is not None:
            n_real = len(self._train_real_ids)
            n_real_total = n_real * (1 + self.bootstrap_n) if self.bootstrap_n > 0 else n_real
            return n_real_total + len(self._train_synthetic_items)
        n = len(self.subject_list)
        if self.train and self.bootstrap_n > 0:
            return n * (1 + self.bootstrap_n)
        return n

    def __getitem__(self, idx):
        if self.train and self._train_real_ids is not None and self._train_synthetic_items is not None:
            n_real = len(self._train_real_ids)
            n_real_total = n_real * (1 + self.bootstrap_n) if self.bootstrap_n > 0 else n_real

            if idx < n_real_total:
                is_bootstrap = idx >= n_real
                actual_idx = idx % n_real if is_bootstrap else idx
                subject = self._train_real_ids[actual_idx]
                timeseries = self._load_timeseries(subject)
                ts_tensor = torch.tensor(timeseries, dtype=torch.float32)
                label = self.behavioral_dict[subject]
                return self._timeseries_to_data(ts_tensor, label, bootstrap=is_bootstrap)
            else:
                syn_idx = idx - n_real_total
                _sid, path, label = self._train_synthetic_items[syn_idx]
                ts = np.load(path).astype(np.float32)
                if ts.ndim == 1:
                    ts = ts.reshape(-1, 1)
                ts = (ts - np.mean(ts, axis=0, keepdims=True)) / (np.std(ts, axis=0, keepdims=True) + 1e-9)
                if self.dynamic_length is not None:
                    if len(ts) >= self.dynamic_length:
                        start = _random.randrange(len(ts) - self.dynamic_length + 1)
                        ts = ts[start : start + self.dynamic_length]
                    else:
                        pad = np.zeros((self.dynamic_length - len(ts), ts.shape[1]), dtype=np.float32)
                        ts = np.vstack([ts, pad])
                ts_tensor = torch.tensor(ts, dtype=torch.float32)
                return self._timeseries_to_data(ts_tensor, label)

        n_base = len(self.subject_list)
        is_bootstrap = self.train and self.bootstrap_n > 0 and idx >= n_base
        actual_idx = idx % n_base if is_bootstrap else idx
        subject = self.subject_list[actual_idx]
        timeseries = self._load_timeseries(subject)
        ts_tensor = torch.tensor(timeseries, dtype=torch.float32)
        label = self.behavioral_dict[subject]
        return self._timeseries_to_data(ts_tensor, label, bootstrap=is_bootstrap)
