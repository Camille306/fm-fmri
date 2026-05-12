"""
Thin shim so that baseline scripts can 'from dataset import HCPRestingFCDataset'
when run from biopoint (with biopoint on sys.path first). Exposes Biopoint data
under the HCP dataset interface for rest-to-task.
"""

from biopoint_dataset import BiopointDatasetAdapter

_DEFAULT_CSV = "./data/biopoint_data.csv"


class HCPRestingFCDataset(BiopointDatasetAdapter):
    """Alias so baselines that expect HCPRestingFCDataset(data_root, task_root, task_name) can load Biopoint."""

    def __init__(
        self,
        data_root: str,
        task_root: str = None,
        task_name: str = None,
        csv_path: str = None,
        **kwargs,
    ):
        super().__init__(
            data_root=data_root,
            csv_path=csv_path or _DEFAULT_CSV,
        )
