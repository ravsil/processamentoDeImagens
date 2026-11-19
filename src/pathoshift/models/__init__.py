"""Models package with classical and deep learning baselines."""

from pathoshift.models.classical import (
    ClassicalBaselineModel,
    HistopathologyFeatureExtractor,
)
try:  # PyTorch é opcional (pipeline final usa apenas CPU/scikit-learn)
    from pathoshift.models.cnn import (
        SimpleCNN,
        build_resnet18,
        get_model,
    )
except ImportError:  # pragma: no cover
    SimpleCNN = None
    build_resnet18 = None
    get_model = None

__all__ = [
    "ClassicalBaselineModel",
    "HistopathologyFeatureExtractor",
    "SimpleCNN",
    "build_resnet18",
    "get_model",
]
