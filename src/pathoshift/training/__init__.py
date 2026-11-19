"""Training and evaluation package."""

from pathoshift.training.metrics import compute_comprehensive_metrics
try:  # PyTorch é opcional (pipeline final usa apenas CPU/scikit-learn)
    from pathoshift.training.trainer import (
        evaluate_model,
        train_deep_model,
        train_epoch,
    )
except ImportError:  # pragma: no cover
    evaluate_model = None
    train_deep_model = None
    train_epoch = None

__all__ = [
    "compute_comprehensive_metrics",
    "evaluate_model",
    "train_deep_model",
    "train_epoch",
]
