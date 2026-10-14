"""Data handling, manifest generation, splits, and PyTorch dataset."""

from pathoshift.data.dataset import PathologyDataset
from pathoshift.data.manifest import (
    build_manifest,
    load_manifest,
    scan_directory,
    validate_manifest,
)
from pathoshift.data.splits import (
    create_patient_aware_splits,
    create_stratified_splits,
)

__all__ = [
    "PathologyDataset",
    "build_manifest",
    "load_manifest",
    "scan_directory",
    "validate_manifest",
    "create_patient_aware_splits",
    "create_stratified_splits",
]
