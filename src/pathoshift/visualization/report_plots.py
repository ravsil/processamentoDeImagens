"""Evaluation visualisations: confusion matrix heatmap, error gallery, and training curves."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

from pathoshift.visualization.eda import set_plotting_style


def plot_confusion_matrix(
    cm: np.ndarray | List[List[float]],
    class_names: List[str],
    output_path: Path | str,
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> None:
    """Plot an annotated confusion matrix heatmap."""
    set_plotting_style()
    cm_arr = np.array(cm, dtype=float)

    if normalize and cm_arr.sum(axis=1, keepdims=True).max() > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_display = cm_arr / cm_arr.sum(axis=1, keepdims=True)
            cm_display = np.nan_to_num(cm_display)
        fmt = ".2%"
    else:
        cm_display = cm_arr
        fmt = ".0f"

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title(title)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_error_gallery(
    records: List[Dict[str, Any]],
    y_true: List[int],
    y_pred: List[int],
    y_prob: np.ndarray,
    class_names: List[str],
    output_path: Path | str,
    n_samples: int = 8,
) -> None:
    """Generate a visual grid of top confident classification errors for model inspection."""
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_prob_arr = np.array(y_prob)

    # Errors: y_true != y_pred
    error_indices = np.where(y_true_arr != y_pred_arr)[0]
    if len(error_indices) == 0:
        return

    # Sort errors by predicted probability (highest confidence mistakes first)
    error_confidences = np.max(y_prob_arr[error_indices], axis=1)
    ranked_order = np.argsort(-error_confidences)
    selected_indices = error_indices[ranked_order[:n_samples]]

    n_cols = 4
    n_rows = int(np.ceil(len(selected_indices) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4.2 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, idx in enumerate(selected_indices):
        rec = records[idx]
        t_label = class_names[y_true_arr[idx]] if y_true_arr[idx] < len(class_names) else str(y_true_arr[idx])
        p_label = class_names[y_pred_arr[idx]] if y_pred_arr[idx] < len(class_names) else str(y_pred_arr[idx])
        conf = error_confidences[ranked_order[i]]

        img_path = Path(rec.get("absolute_path", ""))
        if not img_path.exists() and "relative_path" in rec:
            rel = rec["relative_path"]
            for cand in [Path.cwd() / "data" / rel, Path.cwd() / rel]:
                if cand.exists():
                    img_path = cand
                    break

        try:
            with Image.open(img_path) as img:
                axes[i].imshow(img.convert("RGB"))
        except Exception:
            axes[i].text(0.5, 0.5, "Image unavailable", ha="center", va="center")

        axes[i].set_title(
            f"True: {t_label}\nPred: {p_label}\nConf: {conf:.1%}",
            color="darkred",
            fontsize=10,
        )
        axes[i].axis("off")

    # Turn off unused subplots
    for j in range(len(selected_indices), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Top Confident Model Errors (Inspection Gallery)", fontsize=14, y=0.98)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_training_curves(history: List[Dict[str, Any]], output_path: Path | str) -> None:
    """Plot loss and validation metric trajectories across epochs."""
    set_plotting_style()
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_macro_f1 = [h.get("val_macro_f1", 0.0) for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Loss curve
    axes[0].plot(epochs, train_loss, "o-", label="Train Loss", color="royalblue")
    axes[0].plot(epochs, val_loss, "s--", label="Val Loss", color="crimson")
    axes[0].set_title("Cross-Entropy Loss Trajectory")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Macro F1 curve
    axes[1].plot(epochs, val_macro_f1, "d-", label="Val Macro-F1", color="forestgreen")
    axes[1].set_title("Validation Macro-F1 Score")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
