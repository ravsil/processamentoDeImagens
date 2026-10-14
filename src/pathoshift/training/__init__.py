"""Training and evaluation package."""

from pathoshift.training.metrics import compute_comprehensive_metrics
from pathoshift.training.trainer import (
    evaluate_model,
    train_deep_model,
    train_epoch,
)

__all__ = [
    "compute_comprehensive_metrics",
    "evaluate_model",
    "train_deep_model",
    "train_epoch",
]
