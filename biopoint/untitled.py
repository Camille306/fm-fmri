"""
Windowed dataset for Biopoint rest-to-task: sliding windows (rest segment -> task segment)
with the same batch format as baselines (input, target, subject_id, task_start_idx).
Works with BiopointDatasetAdapter or any object with subject_ids, load_subject, load_task_subject.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class BiopointWindowDataset(Dataset):
    """
    Sliding windows over rest/task pairs. Returns dicts compatible with baseline
    training: input (L,V), target (T,V), subject_id, task_start_idx.
    When use_evs=True and dataset has load_ev_subject, also returns ev (N_events,4) and ev_mask.
    """

    MAX_EV_EVENTS = 64

    def __init__(
        self,
        dataset,  # BiopointDatasetAdapter or any with subject_ids, load_subject, load_task_subject
        lookback_length: int = 200,
        prediction_length: int = 116,
        stride: int = 10,
        normalize: bool = True,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        max_samples_per_subject: int = None,
        norm_sample_size: int = 500,
        norm_batch_size: int = 50,
        use_evs: bool = False,
    ):
        self.dataset = dataset
        self.lookback_length = lookback_length
        self.prediction_length = prediction_length
        self.stride = stride
        self.normalize = normalize
        self.split = split
        self.max_samples_per_subject = max_samples_per_subject
        self.use_evs = use_evs and getattr(dataset, "use_evs", False) and hasattr(dataset, "load_ev_subject")

        self.window_metadata = []
        self.rest_means = None
        self.rest_stds = None
        self.task_means = None
        self.task_stds = None

        self._create_window_indices(train_ratio, val_ratio)

        if self.normalize and len(self.window_metadata) > 0:
            self._compute_normalization_stats(
                sample_size=norm_sample_size, batch_size=norm_batch_size
            )

    def _create_window_indices(self, train_ratio: float, val_ratio: float):
        all_subjects = self.dataset.subject_ids
        n = len(all_subjects)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        if self.split == "train":
            subject_ids = all_subjects[:train_end]
        elif self.split == "val":
            subject_ids = all_subjects[train_end:val_end]
        else:
            subject_ids = all_subjects[val_end:]

        for subject_id in subject_ids:
            try:
                rest_ts = self.dataset.load_subject(subject_id)
                if rest_ts.ndim == 1:
                    rest_ts = rest_ts.reshape(-1, 1)
                R, V = rest_ts.shape

                task_ts = self.dataset.load_task_subject(subject_id)
                if task_ts.ndim == 1:
                    task_ts = task_ts.reshape(-1, 1)
                T, V2 = task_ts.shape
                if V2 != V:
                    continue

                max_rest_idx = R - self.lookback_length
                max_task_idx = T - self.prediction_length
                max_windows = min(max_rest_idx, max_task_idx)
                if max_windows < 0:
                    continue

                wcount = 0
                for rest_start in range(0, max_windows + 1, self.stride):
                    task_start = rest_start
                    if task_start + self.prediction_length > T:
                        break

                    self.window_metadata.append({
                        "subject_id": str(subject_id),
                        "rest_start_idx": int(rest_start),
                        "task_start_idx": int(task_start),
                    })

                    wcount += 1
                    if self.max_samples_per_subject and wcount >= self.max_samples_per_subject:
                        break

            except Exception:
                continue

    def _compute_normalization_stats(
        self, sample_size: int = 500, batch_size: int = 50
    ):
        if len(self.window_metadata) == 0:
            return

        m = min(sample_size, len(self.window_metadata))
        idxs = np.random.choice(len(self.window_metadata), m, replace=False)

        rest_sum = None
        rest_sum_sq = None
        rest_cnt = 0
        task_sum = None
        task_sum_sq = None
        task_cnt = 0
        V = None

        for s in range(0, len(idxs), batch_size):
            batch_idxs = idxs[s : s + batch_size]
            rest_batch = []
            task_batch = []

            for ii in batch_idxs:
                meta = self.window_metadata[ii]
                sid = meta["subject_id"]

                rest = self.dataset.load_subject(sid)
                if rest.ndim == 1:
                    rest = rest.reshape(-1, 1)
                if V is None:
                    V = rest.shape[1]

                rs = meta["rest_start_idx"]
                re = rs + self.lookback_length
                rest_batch.append(rest[rs:re].astype(np.float32))

                task = self.dataset.load_task_subject(sid)
                if task.ndim == 1:
                    task = task.reshape(-1, 1)
                ts = meta["task_start_idx"]
                te = ts + self.prediction_length
                task_batch.append(task[ts:te].astype(np.float32))

            if rest_batch:
                r = np.stack(rest_batch).reshape(-1, V)
                if rest_sum is None:
                    rest_sum = r.sum(0)
                    rest_sum_sq = (r ** 2).sum(0)
                else:
                    rest_sum += r.sum(0)
                    rest_sum_sq += (r ** 2).sum(0)
                rest_cnt += r.shape[0]

            if task_batch:
                t = np.stack(task_batch).reshape(-1, V)
                if task_sum is None:
                    task_sum = t.sum(0)
                    task_sum_sq = (t ** 2).sum(0)
                else:
                    task_sum += t.sum(0)
                    task_sum_sq += (t ** 2).sum(0)
                task_cnt += t.shape[0]

        self.rest_means = rest_sum / max(rest_cnt, 1)
        rest_var = (rest_sum_sq / max(rest_cnt, 1)) - self.rest_means ** 2
        self.rest_stds = np.sqrt(np.maximum(rest_var, 0.0))
        self.rest_stds = np.where(self.rest_stds < 1e-8, 1.0, self.rest_stds)

        self.task_means = task_sum / max(task_cnt, 1)
        task_var = (task_sum_sq / max(task_cnt, 1)) - self.task_means ** 2
        self.task_stds = np.sqrt(np.maximum(task_var, 0.0))
        self.task_stds = np.where(self.task_stds < 1e-8, 1.0, self.task_stds)

    def __len__(self):
        return len(self.window_metadata)

    def __getitem__(self, idx):
        meta = self.window_metadata[idx]
        sid = meta["subject_id"]

        rest = self.dataset.load_subject(sid)
        if rest.ndim == 1:
            rest = rest.reshape(-1, 1)
        rs = meta["rest_start_idx"]
        re = rs + self.lookback_length
        x = rest[rs:re].astype(np.float32)

        task = self.dataset.load_task_subject(sid)
        if task.ndim == 1:
            task = task.reshape(-1, 1)
        ts = meta["task_start_idx"]
        te = ts + self.prediction_length
        y = task[ts:te].astype(np.float32)

        if self.normalize and self.rest_means is not None:
            x = (x - self.rest_means) / self.rest_stds
            y = (y - self.task_means) / self.task_stds

        out = {
            "input": torch.from_numpy(x),
            "target": torch.from_numpy(y),
            "subject_id": sid,
            "task_start_idx": int(ts),
        }

        if self.use_evs:
            try:
                ev_full = self.dataset.load_ev_subject(sid)  # (N_events, 4)
                N_ev, n_cols = ev_full.shape
                if N_ev > self.MAX_EV_EVENTS:
                    ev_full = ev_full[: self.MAX_EV_EVENTS]
                    N_ev = self.MAX_EV_EVENTS
                pad_len = self.MAX_EV_EVENTS - N_ev
                if pad_len > 0:
                    ev_full = np.concatenate(
                        [ev_full, np.zeros((pad_len, n_cols), dtype=np.float32)], axis=0
                    )
                ev_mask = np.zeros(self.MAX_EV_EVENTS, dtype=np.float32)
                ev_mask[:N_ev] = 1.0
                out["ev"] = torch.from_numpy(ev_full)
                out["ev_mask"] = torch.from_numpy(ev_mask)
            except (FileNotFoundError, ValueError):
                out["ev"] = torch.zeros(self.MAX_EV_EVENTS, 4, dtype=torch.float32)
                out["ev_mask"] = torch.zeros(self.MAX_EV_EVENTS, dtype=torch.float32)
        else:
            out["ev"] = torch.zeros(self.MAX_EV_EVENTS, 4, dtype=torch.float32)
            out["ev_mask"] = torch.zeros(self.MAX_EV_EVENTS, dtype=torch.float32)

        return out
