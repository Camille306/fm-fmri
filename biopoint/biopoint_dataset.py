import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _load_roi_ts_from_pt(path: str, transpose_if_first_smaller_than_second: bool = True) -> np.ndarray:
    # Helper for loading DK atlas ROI time-series from a `.pt` file.
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


class fMRI_Biopoint_Dataset(Dataset):
    """Dataset that returns the matching (rest, task) pair per subject."""

    def __init__(self, n_max_time_step=316, data_root="./data/biopoint_data"):
        self.patient_df = pd.read_csv(
            "./data/biopoint_data.csv"
        )
        self.n_max_time_step = n_max_time_step
        self.data_root = data_root

        # Keep only subjects that have BOTH rest and task time series (matched)
        valid_indices = []
        for idx in range(len(self.patient_df)):
            sub_id = str(self.patient_df.iloc[idx]["subject_id"])
            rest_path = os.path.join(
                self.data_root, "output", sub_id, "rest", f"{sub_id}_shen268_ts.npy"
            )
            task_path = os.path.join(
                self.data_root, "output", sub_id, "task", f"{sub_id}_shen268_ts.npy"
            )
            if os.path.exists(rest_path) and os.path.exists(task_path):
                valid_indices.append(idx)

        self.patient_df = self.patient_df.iloc[valid_indices].reset_index(drop=True)
        print(
            f"Matched dataset: {len(self.patient_df)} subjects with both rest and task time series"
        )

    def __len__(self):
        return len(self.patient_df)

    def normalize(self, time_series):
        return (time_series - np.mean(time_series)) / (np.std(time_series) + 1e-4)

    def _load_and_pad(self, time_series):
        """Normalize and pad a single time series to n_max_time_step."""
        normalized = np.zeros(time_series.shape)
        for i in range(time_series.shape[1]):
            normalized[:, i] = self.normalize(time_series[:, i])
        pad_len = self.n_max_time_step - time_series.shape[0]
        padded = np.vstack((normalized, np.zeros((pad_len, time_series.shape[1]))))
        return torch.from_numpy(padded).float()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sub_id = str(self.patient_df.iloc[idx]["subject_id"])
        rest_path = os.path.join(
            self.data_root, "output", sub_id, "rest", f"{sub_id}_shen268_ts.npy"
        )
        task_path = os.path.join(
            self.data_root, "output", sub_id, "task", f"{sub_id}_shen268_ts.npy"
        )

        if not os.path.exists(rest_path):
            raise FileNotFoundError(f"Rest time series not found: {rest_path}")
        if not os.path.exists(task_path):
            raise FileNotFoundError(f"Task time series not found: {task_path}")

        rest_ts = np.load(rest_path)
        task_ts = np.load(task_path)

        rest_padded = self._load_and_pad(rest_ts)
        task_padded = self._load_and_pad(task_ts)

        group = self.patient_df.iloc[idx]["group"]
        label = 1 if group == "pat" else 0

        # Matching pair: (rest, task, label) for the same subject
        return rest_padded, task_padded, torch.tensor(label, dtype=torch.long)


def _load_eprime_ev_file(path: str) -> np.ndarray:
    """Load one EV file (onset_sec, duration_sec, amplitude). Returns (N, 3) float."""
    arr = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] < 2:
        arr = np.hstack([arr, np.zeros((arr.shape[0], 2 - arr.shape[1]), dtype=np.float64)])
    if arr.shape[1] < 3:
        arr = np.hstack([arr, np.ones((arr.shape[0], 1), dtype=np.float64)])
    return arr.astype(np.float32)


class BiopointDatasetAdapter:
    """
    Adapter that exposes the same interface as HCPRestingFCDataset (subject_ids,
    load_subject, load_task_subject) so baseline windowed datasets and training
    scripts can use Biopoint data. Returns raw numpy (T, V), no padding.
    When ev_root is set, also filters to subjects with EV files and provides load_ev_subject().
    """

    # Eprime condition codes: BioPoints_Timing = 1, RandPoints_Timing = 2
    BIOPOINTS_CONDITION = 1
    RANDPOINTS_CONDITION = 2

    def __init__(
        self,
        data_root: str = "./data/biopoint_data",
        csv_path: str = "./data/biopoint_data.csv",
        ev_root: str = None,
        ev_time_scale_tr: float = None,
        atlas_source: str = "dk",
        dk_atlas_ts_root: str = "./data/biopoint_dk_atlas",
        dk_rest_filename_template: str = "{subject_id}_rest_roi_ts.pt",
        dk_task_filename_template: str = "{subject_id}_task_roi_ts.pt",
        shen268_filename_template: str = "{subject_id}_shen268_ts.npy",
        pt_transpose_if_first_smaller_than_second: bool = True,
    ):
        self.data_root = data_root
        self.task_root = data_root  # so use_task_target=True in window datasets
        self.ev_root = os.path.abspath(ev_root) if ev_root else None
        self.use_evs = self.ev_root is not None
        self.ev_time_scale_tr = ev_time_scale_tr  # None = infer from task length when first loading EV
        self.atlas_source = atlas_source
        self.dk_atlas_ts_root = dk_atlas_ts_root or "./data/biopoint_dk_atlas"
        self.dk_rest_filename_template = dk_rest_filename_template
        self.dk_task_filename_template = dk_task_filename_template
        self.shen268_filename_template = shen268_filename_template
        self.pt_transpose_if_first_smaller_than_second = pt_transpose_if_first_smaller_than_second

        self.patient_df = pd.read_csv(csv_path)
        n_csv = len(self.patient_df)
        valid_indices = []
        n_with_roi_ts = 0
        n_roi_ts_but_no_ev = 0
        first_sub = str(self.patient_df.iloc[0]["subject_id"]) if n_csv else None
        first_rest, first_task = (
            self._subject_rest_task_paths(first_sub) if first_sub is not None else (None, None)
        )

        for idx in range(n_csv):
            sub_id = str(self.patient_df.iloc[idx]["subject_id"])
            rest_path, task_path = self._subject_rest_task_paths(sub_id)
            if not (os.path.exists(rest_path) and os.path.exists(task_path)):
                continue
            n_with_roi_ts += 1
            if self.use_evs:
                if not self._has_ev(sub_id):
                    n_roi_ts_but_no_ev += 1
                    continue
            valid_indices.append(idx)

        self.patient_df = self.patient_df.iloc[valid_indices].reset_index(drop=True)
        self._subject_ids = self.patient_df["subject_id"].astype(str).tolist()

        if len(self._subject_ids) == 0 and n_csv > 0:
            print(
                "Biopoint adapter: 0 subjects after filtering.\n"
                f"  csv_path={csv_path!r}  rows={n_csv}\n"
                f"  atlas_source={self.atlas_source!r}\n"
                + (
                    f"  dk_atlas_ts_root={self.dk_atlas_ts_root!r}\n"
                    f"  example (first CSV subject {first_sub!r}):\n"
                    f"    rest -> {first_rest!r} exists={os.path.exists(first_rest)}\n"
                    f"    task -> {first_task!r} exists={os.path.exists(first_task)}\n"
                    if self.atlas_source == "dk"
                    else f"  data_root={self.data_root!r}\n"
                )
                + (
                    f"  subjects_with_both_roi_ts_files={n_with_roi_ts}\n"
                    f"  ev_root={self.ev_root!r}\n"
                    f"  expected EV layout: ev_root/<subject_id>/eprime_timing/BioPoints_Timing "
                    f"and RandPoints_Timing (or same files directly under ev_root/<subject_id>/)\n"
                    f"  subjects_with_roi_ts_but_missing_ev_pair={n_roi_ts_but_no_ev}\n"
                    if self.use_evs
                    else ""
                )
                + "  Fix: set --dk_atlas_ts_root to where *_rest_roi_ts.pt / *_task_roi_ts.pt live, "
                "or fix --eprime_root / folder names, or use --atlas_source shen268 if you only have .npy.\n"
            )
        elif self.use_evs and n_csv > 0:
            print(
                f"Biopoint adapter filter: csv_rows={n_csv}  with_roi_ts={n_with_roi_ts}  "
                f"with_roi_ts_and_ev={len(self._subject_ids)}"
            )

        if self.use_evs:
            if self.ev_time_scale_tr is None and len(self._subject_ids) > 0:
                task = self.load_task_subject(self._subject_ids[0])
                n_tr = int(task.shape[0])
                self.ev_time_scale_tr = max(n_tr * 0.72, 1.0)
            print(f"Biopoint adapter (with EV): {len(self._subject_ids)} subjects with rest, task, and eprime EV files")
        else:
            print(f"Biopoint adapter: {len(self._subject_ids)} subjects using atlas_source={self.atlas_source!r}")

    def _subject_rest_task_paths(self, subject_id: str) -> tuple[str, str]:
        """Return (rest_ts_path, task_ts_path) for the configured atlas selection."""
        subject_id = str(subject_id)

        if self.atlas_source == "shen268":
            fname = self.shen268_filename_template.format(subject_id=subject_id)
            rest_path = os.path.join(self.data_root, "output", subject_id, "rest", fname)
            task_path = os.path.join(self.data_root, "output", subject_id, "task", fname)
            return rest_path, task_path

        if self.atlas_source == "dk":
            rest_fname = self.dk_rest_filename_template.format(subject_id=subject_id)
            task_fname = self.dk_task_filename_template.format(subject_id=subject_id)
            rest_path = os.path.join(self.dk_atlas_ts_root, rest_fname)
            task_path = os.path.join(self.dk_atlas_ts_root, task_fname)
            return rest_path, task_path

        raise ValueError(f"Unknown atlas_source={self.atlas_source!r}. Expected one of: 'dk', 'shen268'.")

    @property
    def subject_ids(self):
        return self._subject_ids

    def _ev_paths(self, subject_id: str):
        """Return (path_bio, path_rand) for BioPoints_Timing and RandPoints_Timing. Tries eprime_timing subdir then direct."""
        s = str(subject_id)
        base = os.path.join(self.ev_root, s)
        subdir = os.path.join(base, "eprime_timing")
        for folder in (subdir, base):
            bio = os.path.join(folder, "BioPoints_Timing")
            rand = os.path.join(folder, "RandPoints_Timing")
            if os.path.isfile(bio) and os.path.isfile(rand):
                return bio, rand
        return None, None

    def _has_ev(self, subject_id: str) -> bool:
        bio, rand = self._ev_paths(subject_id)
        return bio is not None

    def load_ev_subject(self, subject_id: str) -> np.ndarray:
        """
        Load paired EV files: BioPoints_Timing (condition 1) and RandPoints_Timing (condition 2).
        Paths: eprime_root/{subject_id}/eprime_timing/BioPoints_Timing and RandPoints_Timing.
        Returns (N, 4) float32: onset_norm, duration_norm, amplitude (z-scored), condition (1 or 2).
        """
        subject_id = str(subject_id)
        bio_path, rand_path = self._ev_paths(subject_id)
        if bio_path is None:
            raise FileNotFoundError(
                f"EV files not found for {subject_id} under {self.ev_root}/{subject_id}/[eprime_timing/]BioPoints_Timing and RandPoints_Timing"
            )

        blocks = []
        for path, cond in [(bio_path, self.BIOPOINTS_CONDITION), (rand_path, self.RANDPOINTS_CONDITION)]:
            arr = _load_eprime_ev_file(path)  # (N, 3): onset, duration, amplitude
            cond_col = np.full((arr.shape[0], 1), float(cond), dtype=np.float32)
            blocks.append(np.hstack([arr, cond_col]))

        ev = np.vstack(blocks).astype(np.float32)

        time_scale = float(self.ev_time_scale_tr or 1.0)
        if ev.shape[1] >= 1:
            ev[:, 0] = ev[:, 0].astype(np.float64) / time_scale
        if ev.shape[1] >= 2:
            ev[:, 1] = ev[:, 1].astype(np.float64) / time_scale
        if ev.shape[1] >= 3:
            a = ev[:, 2].astype(np.float64)
            s = np.std(a)
            ev[:, 2] = (a - np.mean(a)) / (s if s > 1e-12 else 1.0)
        return ev.astype(np.float32)

    def load_subject(self, subject_id: str) -> np.ndarray:
        subject_id = str(subject_id)
        rest_path, _ = self._subject_rest_task_paths(subject_id)
        if not os.path.exists(rest_path):
            raise FileNotFoundError(f"Rest not found: {rest_path}")

        if self.atlas_source == "shen268":
            x = np.load(rest_path).astype(np.float32)
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            return x

        # DK atlas stored as `.pt`.
        return _load_roi_ts_from_pt(
            rest_path,
            transpose_if_first_smaller_than_second=self.pt_transpose_if_first_smaller_than_second,
        )

    def load_task_subject(self, subject_id: str) -> np.ndarray:
        subject_id = str(subject_id)
        _, task_path = self._subject_rest_task_paths(subject_id)
        if not os.path.exists(task_path):
            raise FileNotFoundError(f"Task not found: {task_path}")

        if self.atlas_source == "shen268":
            x = np.load(task_path).astype(np.float32)
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            return x

        return _load_roi_ts_from_pt(
            task_path,
            transpose_if_first_smaller_than_second=self.pt_transpose_if_first_smaller_than_second,
        )
