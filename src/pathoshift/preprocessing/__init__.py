"""Histopathology image preprocessing and quality package."""

from pathoshift.preprocessing.quality import (
    compute_tissue_mask,
    compute_focus_measure,
    evaluate_tile_quality,
)
from pathoshift.preprocessing.stain_norm import (
    MacenkoNormalizer,
    ReinhardNormalizer,
)
try:  # PyTorch é opcional (pipeline final usa apenas CPU/scikit-learn)
    from pathoshift.preprocessing.transforms import (
        StainNormTransform,
        get_image_transforms,
    )
except ImportError:  # pragma: no cover
    StainNormTransform = None
    get_image_transforms = None

__all__ = [
    "compute_tissue_mask",
    "compute_focus_measure",
    "evaluate_tile_quality",
    "MacenkoNormalizer",
    "ReinhardNormalizer",
    "StainNormTransform",
    "get_image_transforms",
]
