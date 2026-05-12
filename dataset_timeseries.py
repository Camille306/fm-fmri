import os
import numpy as np
from pathlib import Path
from typing import Dict, Optional


class HCPRestingFCDataset:
    """
    Dataset loader for HCP resting-state functional connectivity data.
    
    Loads REST1RL_Shen268_ts.npy files from subject folders in the HCP resting FC directory.
    """
    
    def __init__(self, data_root: str = "./data/hcp-resting-fc"):
        """
        Initialize the dataset.
        
        Args:
            data_root: Root directory containing subject folders
        """
        self.data_root = Path(data_root)
        self.subject_ids = self._get_subject_ids()
        
    def _get_subject_ids(self) -> list:
        """Get all subject IDs from the data root directory."""
        if not self.data_root.exists():
            raise ValueError(f"Data root directory does not exist: {self.data_root}")
        
        subject_ids = []
        for item in self.data_root.iterdir():
            if item.is_dir():
                subject_ids.append(item.name)
        
        return sorted(subject_ids)
    
    def get_subject_path(self, subject_id: str) -> Path:
        """
        Get the path to a subject's timeseries file.
        
        Args:
            subject_id: Subject ID string
            
        Returns:
            Path to REST1RL_Shen268_ts.npy file
        """
        return self.data_root / subject_id / "timeseries" / "REST1RL_Shen268_ts.npy"
    
    def load_subject(self, subject_id: str) -> np.ndarray:
        """
        Load timeseries data for a specific subject.
        
        Args:
            subject_id: Subject ID string
            
        Returns:
            numpy array containing the timeseries data
            
        Raises:
            FileNotFoundError: If the file doesn't exist for this subject
        """
        file_path = self.get_subject_path(subject_id)
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Timeseries file not found for subject {subject_id}: {file_path}"
            )
        
        return np.load(file_path)
    
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
            Tuple of (subject_id, timeseries_array)
        """
        subject_id = self.subject_ids[idx]
        return subject_id, self.load_subject(subject_id)


if __name__ == "__main__":
    # Example usage
    dataset = HCPRestingFCDataset()
    
    print(f"Found {len(dataset)} subjects")
    print(f"Subject IDs: {dataset.subject_ids[:10]}...")  # Show first 10
    
    # Load a single subject
    if len(dataset) > 0:
        first_subject = dataset.subject_ids[0]
        print(f"\nLoading subject: {first_subject}")
        timeseries = dataset.load_subject(first_subject)
        print(f"Timeseries shape: {timeseries.shape}")

