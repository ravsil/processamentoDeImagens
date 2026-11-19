"""Data handling, manifest generation, splits, and PyTorch dataset."""

try:  # PyTorch é opcional
    from pathoshift.data.dataset import PathologyDataset
except ImportError:  # pragma: no cover
    PathologyDataset = None
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
