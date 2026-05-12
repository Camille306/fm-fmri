import os
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch


class HCPRestingFCDataset:
    """
    Dataset loader for HCP resting-state + task timeseries data.
    
    Loads resting-state timeseries files from subject folders in the HCP resting FC directory.
    This repo is currently configured to use **AAL3** resting-state timeseries:
    - `REST1_LR_AAL3_ts.npy`
    so that the ROI dimension matches the task data parcellation.
    """
    
    def __init__(
        self, 
        data_root: str = "./data/hcp-resting-fc",
        task_root: Optional[str] = None,
        task_name: str = "emotion"
    ):
        """
        Initialize the dataset.
        
        Args:
            data_root: Root directory containing subject folders for resting state data
            task_root: Root directory for task data (if None, only resting data will be loaded)
            task_name: Name of the task (e.g., "emotion")
        """
        self.data_root = Path(data_root)
        self.task_root = Path(task_root) if task_root else None
        self.task_name = task_name
        self.subject_ids = self._get_subject_ids()
        
        # Filter to only subjects that have both resting and task data if task_root is provided
        if self.task_root:
            self.subject_ids = self._filter_paired_subjects()
        
    def _get_subject_ids(self) -> list:
        """Get all subject IDs from the data root directory."""
        if not self.data_root.exists():
            raise ValueError(f"Data root directory does not exist: {self.data_root}")
        
        subject_ids = []
        for item in self.data_root.iterdir():
            if item.is_dir():
                subject_ids.append(item.name)
        
        return sorted(subject_ids)
    
    def _filter_paired_subjects(self) -> list:
        """Filter to only subjects that have both resting and task data."""
        if not self.task_root or not self.task_root.exists():
            return self.subject_ids
        
        paired_subjects = []
        for subject_id in self.subject_ids:
            rest_path = self.get_subject_path(subject_id)
            task_path = self.get_task_path(subject_id)
            
            if rest_path.exists() and task_path.exists():
                paired_subjects.append(subject_id)
        
        return sorted(paired_subjects)
    
    def get_subject_path(self, subject_id: str) -> Path:
        """
        Get the path to a subject's resting state timeseries file.
        
        Args:
            subject_id: Subject ID string
            
        Returns:
            Path to resting-state timeseries file (currently `REST1_LR_AAL3_ts.npy`)
        """
        # Use AAL3 parcellation to match task ROI dimensionality.
        return self.data_root / subject_id / "timeseries" / "REST1_LR_AAL3_ts.npy"
    
    def get_task_path(self, subject_id: str) -> Path:
        """
        Get the path to a subject's task timeseries file.
        
        Args:
            subject_id: Subject ID string
            
        Returns:
            Path to task timeseries file
            
        The method tries multiple patterns:
        1. {task_root}/{task_name}/roi_data_std/{subject_id} (as file or directory)
        2. {task_root}/{task_name}/roi_data_std/{subject_id}.{ext} (with extensions)
        """
        if not self.task_root:
            raise ValueError("Task root not specified")
        
        parent_dir = self.task_root / self.task_name / "roi_data_std"
        task_path = parent_dir / subject_id
        
        # Pattern 1: Check if {subject_id} is a file directly (no extension)
        if task_path.exists() and task_path.is_file():
            return task_path
        
        # Pattern 2: Check if {subject_id} is a directory
        if task_path.exists() and task_path.is_dir():
            # Try files inside the directory with common extensions
            for ext in ['.pt', '.pth', '.npy']:
                candidate = task_path / f"{subject_id}{ext}"
                if candidate.exists():
                    return candidate
            # Or a file named exactly {subject_id} inside the directory
            candidate = task_path / subject_id
            if candidate.exists():
                return candidate
        
        # Pattern 3: Check if file exists with extensions in parent directory
        for ext in ['.pt', '.pth', '.npy']:
            candidate = parent_dir / f"{subject_id}{ext}"
            if candidate.exists():
                return candidate
        
        # Pattern 4: Check if file exists without extension in parent directory
        if parent_dir.exists() and (parent_dir / subject_id).exists():
            return parent_dir / subject_id
        
        # Default: return the expected path (will raise error in load_task_subject if doesn't exist)
        return task_path
    
    def load_subject(self, subject_id: str) -> np.ndarray:
        """
        Load resting state timeseries data for a specific subject.
        
        Args:
            subject_id: Subject ID string
            
        Returns:
            numpy array containing the resting state timeseries data
            
        Raises:
            FileNotFoundError: If the file doesn't exist for this subject
        """
        file_path = self.get_subject_path(subject_id)
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Resting state timeseries file not found for subject {subject_id}: {file_path}"
            )
        
        return np.load(file_path)
    
    def load_task_subject(self, subject_id: str) -> np.ndarray:
        """
        Load task timeseries data for a specific subject.
        
        Args:
            subject_id: Subject ID string
            
        Returns:
            numpy array containing the task timeseries data
            
        Raises:
            FileNotFoundError: If the file doesn't exist for this subject
        """
        if not self.task_root:
            raise ValueError("Task root not specified")
        
        file_path = self.get_task_path(subject_id)
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Task timeseries file not found for subject {subject_id}: {file_path}"
            )
        
        # Try loading as PyTorch first (common format for task data)
        if file_path.suffix in ['.pt', '.pth'] or file_path.suffix == '':
            try:
                task_data = torch.load(str(file_path), map_location='cpu')
                # Convert to numpy if it's a tensor
                if isinstance(task_data, torch.Tensor):
                    return task_data.numpy()
                elif isinstance(task_data, np.ndarray):
                    return task_data
                elif isinstance(task_data, dict):
                    # If it's a dict, try to extract the data
                    if 'data' in task_data:
                        data = task_data['data']
                        return data.numpy() if isinstance(data, torch.Tensor) else data
                    else:
                        # Take the first tensor/array value
                        for v in task_data.values():
                            if isinstance(v, (torch.Tensor, np.ndarray)):
                                return v.numpy() if isinstance(v, torch.Tensor) else v
                        raise ValueError(f"Could not extract array from dict: {list(task_data.keys())}")
                else:
                    raise ValueError(f"Unexpected data type: {type(task_data)}")
            except Exception as e:
                # If PyTorch loading fails, try NumPy
                try:
                    return np.load(str(file_path))
                except:
                    raise FileNotFoundError(
                        f"Could not load task data for subject {subject_id} from {file_path}: {e}"
                    )
        else:
            # Try NumPy loading
            return np.load(str(file_path))
    
    def load_all_subjects(self) -> Dict[str, np.ndarray]:
        """
        Load timeseries data for all subjects.
        
        Returns:
            Dictionary mapping subject IDs to their timeseries arrays
        """
        data = {}
        for subject_id in self.subject_ids:
            try:
                data[subject_id] = self.load_subject(subject_id)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue
        
        return data
    
    def __len__(self) -> int:
        """Return the number of subjects in the dataset."""
        return len(self.subject_ids)
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Get a subject by index.
        
        Args:
            idx: Index of the subject
            
        Returns:
            If task_root is provided: Tuple of (subject_id, rest_timeseries, task_timeseries)
            Otherwise: Tuple of (subject_id, rest_timeseries)
        """
        subject_id = self.subject_ids[idx]
        rest_data = self.load_subject(subject_id)
        
        if self.task_root:
            task_data = self.load_task_subject(subject_id)
            return subject_id, rest_data, task_data
        else:
            return subject_id, rest_data


if __name__ == "__main__":
    # Example usage
    task_root = "./data/hcp-task-ts"
    dataset = HCPRestingFCDataset(task_root=task_root, task_name="emotion")
    
    print(f"Found {len(dataset)} subjects with paired resting and task data")
    print(f"Subject IDs: {dataset.subject_ids[:10]}...")  # Show first 10
    
    # Load a single subject
    if len(dataset) > 0:
        first_subject = dataset.subject_ids[0]
        print(f"\nLoading subject: {first_subject}")
        rest_timeseries = dataset.load_subject(first_subject)
        print(f"Resting timeseries shape: {rest_timeseries.shape}")
        
        if dataset.task_root:
            task_timeseries = dataset.load_task_subject(first_subject)
            print(f"Task timeseries shape: {task_timeseries.shape}")
            
            # Test __getitem__
            subject_id, rest_data, task_data = dataset[0]
            print(f"\n__getitem__ test:")
            print(f"  Subject ID: {subject_id}")
            print(f"  Rest data shape: {rest_data.shape}")
            print(f"  Task data shape: {task_data.shape}")

