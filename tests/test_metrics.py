"""Unit tests for metrics and evaluation logic."""

import numpy as np
import pytest

from pathoshift.training.metrics import compute_comprehensive_metrics


def test_metrics_perfect_predictions():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 2, 2]
    class_names = ["tumor", "stroma", "normal"]

    metrics = compute_comprehensive_metrics(y_true, y_pred, class_names=class_names)

    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro"]["f1"] == 1.0
    assert metrics["per_class"]["tumor"]["precision"] == 1.0
    assert metrics["per_class"]["stroma"]["recall"] == 1.0


def test_metrics_imbalanced_classes():
    y_true = [0, 0, 0, 0, 1]
    y_pred = [0, 0, 0, 0, 0]  # Trivial majority class baseline
    class_names = ["normal", "rare_lesion"]

    metrics = compute_comprehensive_metrics(y_true, y_pred, class_names=class_names)

    # Accuracy is high (0.8), but balanced accuracy reveals failure on rare class (0.5)
    assert metrics["accuracy"] == 0.8
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["per_class"]["rare_lesion"]["f1"] == 0.0
