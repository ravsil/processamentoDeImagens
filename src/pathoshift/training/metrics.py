"""Evaluation metrics: per-class, macro, balanced accuracy, and confusion matrix."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_comprehensive_metrics(
    y_true: List[int] | np.ndarray,
    y_pred: List[int] | np.ndarray,
    class_names: Optional[List[str]] = None,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Calculate complete metric suite for histopathology evaluation.

    Includes per-class precision/recall/F1, macro/weighted averages,
    balanced accuracy, raw and row-normalized confusion matrix.

    Args:
        y_true: Ground truth integer class labels.
        y_pred: Predicted integer class labels.
        class_names: Optional list of human-readable class names.
        y_prob: Optional prediction probability array (N, num_classes).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    if class_names is None:
        class_names = [f"class_{c}" for c in unique_classes]

    # Global accuracy & balanced accuracy
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    # Macro and weighted metrics
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Per-class metrics
    class_p, class_r, class_f1, class_supp = precision_recall_fscore_support(
        y_true, y_pred, labels=unique_classes, zero_division=0
    )

    per_class_dict: Dict[str, Dict[str, float]] = {}
    for idx, c in enumerate(unique_classes):
        name = class_names[c] if c < len(class_names) else f"class_{c}"
        per_class_dict[name] = {
            "precision": float(class_p[idx]),
            "recall": float(class_r[idx]),
            "f1": float(class_f1[idx]),
            "support": int(class_supp[idx]),
        }

    # Confusion Matrix
    cm_raw = confusion_matrix(y_true, y_pred, labels=unique_classes)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

    result: Dict[str, Any] = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro": {
            "precision": float(macro_p),
            "recall": float(macro_r),
            "f1": float(macro_f1),
        },
        "weighted": {
            "precision": float(weighted_p),
            "recall": float(weighted_r),
            "f1": float(weighted_f1),
        },
        "per_class": per_class_dict,
        "confusion_matrix": cm_raw.tolist(),
        "confusion_matrix_normalized": [[round(val, 4) for val in row] for row in cm_norm.tolist()],
        "classes": [class_names[c] if c < len(class_names) else f"class_{c}" for c in unique_classes],
    }

    # Confidence statistics if probabilities provided
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        confidences = np.max(y_prob, axis=1)
        correct_mask = (y_true == y_pred)

        result["confidence"] = {
            "mean_confidence_overall": float(np.mean(confidences)),
            "mean_confidence_correct": float(np.mean(confidences[correct_mask])) if np.any(correct_mask) else 0.0,
            "mean_confidence_incorrect": float(np.mean(confidences[~correct_mask])) if np.any(~correct_mask) else 0.0,
        }

    return result
