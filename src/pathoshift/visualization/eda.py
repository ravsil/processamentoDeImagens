"""Exploratory Data Analysis (EDA) visualisations and report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

from pathoshift.data.manifest import load_manifest


def set_plotting_style():
    """Configure clean publication-ready plotting aesthetics."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.autolayout": True,
    })


def plot_class_distribution(records: List[Dict[str, Any]], output_path: Path | str) -> None:
    """Plot class frequency broken down by dataset split."""
    set_plotting_style()
    plt.figure(figsize=(9, 5))

    classes = [r.get("label", "unknown") for r in records]
    splits = [r.get("split", "unassigned") for r in records]

    df_data = {"Class": classes, "Split": splits}
    import pandas as pd
    df = pd.DataFrame(df_data)

    ax = sns.countplot(data=df, x="Class", hue="Split")
    plt.title("Histopathology Dataset Class Distribution Across Splits")
    plt.xlabel("Class Category")
    plt.ylabel("Tile Count")
    plt.xticks(rotation=25, ha="right")

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{int(height)}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_color_channels(records: List[Dict[str, Any]], output_path: Path | str, max_samples: int = 150) -> None:
    """Plot color channel intensity distributions (RGB and HSV)."""
    set_plotting_style()
    sample_records = records[:max_samples]

    r_vals, g_vals, b_vals = [], [], []
    h_vals, s_vals, v_vals = [], [], []

    for r in sample_records:
        path = r.get("absolute_path", "")
        if not Path(path).exists() and "relative_path" in r:
            rel = r["relative_path"]
            for cand in [Path.cwd() / "data" / rel, Path.cwd() / rel]:
                if cand.exists():
                    path = str(cand)
                    break
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        # Sample 500 pixels per image to keep memory low
        step = max(1, (rgb.shape[0] * rgb.shape[1]) // 500)
        rgb_flat = rgb.reshape(-1, 3)[::step]
        hsv_flat = hsv.reshape(-1, 3)[::step]

        r_vals.extend(rgb_flat[:, 0])
        g_vals.extend(rgb_flat[:, 1])
        b_vals.extend(rgb_flat[:, 2])

        h_vals.extend(hsv_flat[:, 0])
        s_vals.extend(hsv_flat[:, 1])
        v_vals.extend(hsv_flat[:, 2])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # RGB distribution
    for vals, col, lbl in [(r_vals, "red", "Red"), (g_vals, "green", "Green"), (b_vals, "blue", "Blue")]:
        if len(vals) > 0:
            sns.histplot(vals, ax=axes[0], color=col, label=lbl, kde=(np.var(vals) > 1e-4), stat="density", element="step", alpha=0.3)
    axes[0].set_title("RGB Channel Intensity Distribution")
    axes[0].set_xlabel("Pixel Value [0-255]")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # HSV distribution
    for vals, col, lbl in [(h_vals, "purple", "Hue"), (s_vals, "orange", "Saturation"), (v_vals, "gray", "Value")]:
        if len(vals) > 0:
            sns.histplot(vals, ax=axes[1], color=col, label=lbl, kde=(np.var(vals) > 1e-4), stat="density", element="step", alpha=0.3)
    axes[1].set_title("HSV Channel Distribution")
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_quality_metrics(records: List[Dict[str, Any]], output_path: Path | str) -> None:
    """Plot tissue ratio and focus sharpness distributions."""
    set_plotting_style()
    import pandas as pd

    df = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Tissue ratio
    if "tissue_ratio" in df.columns:
        kde_flag = df["tissue_ratio"].nunique() > 2
        sns.histplot(
            data=df,
            x="tissue_ratio",
            hue="label" if "label" in df.columns else None,
            kde=kde_flag,
            ax=axes[0],
            bins=15,
        )
        axes[0].axvline(0.40, color="crimson", linestyle="--", label="Rejection Threshold (0.40)")
        axes[0].set_title("Tissue Area Ratio Distribution")
        axes[0].set_xlabel("Tissue Fraction [0.0 - 1.0]")
        axes[0].legend()

    # Focus score (Laplacian variance)
    if "focus_score" in df.columns:
        sns.boxplot(data=df, x="label" if "label" in df.columns else None, y="focus_score", ax=axes[1])
        axes[1].set_yscale("log")
        axes[1].axhline(50.0, color="crimson", linestyle="--", label="Min Focus Threshold")
        axes[1].set_title("Tile Focus Score (Laplacian Variance, log scale)")
        axes[1].set_ylabel("Laplacian Variance")
        axes[1].legend()

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_full_eda(manifest_path: str | Path, output_dir: str | Path) -> Dict[str, Any]:
    """Execute complete automated EDA suite and export figures and JSON summary."""
    records = load_manifest(manifest_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_class_distribution(records, out_dir / "eda_class_distribution.png")
    plot_color_channels(records, out_dir / "eda_color_channels.png")
    plot_quality_metrics(records, out_dir / "eda_quality_metrics.png")

    # Compute numerical summary
    widths = [r.get("width", 0) for r in records]
    heights = [r.get("height", 0) for r in records]
    tissue_ratios = [r.get("tissue_ratio", 1.0) for r in records]
    focus_scores = [r.get("focus_score", 100.0) for r in records]

    labels = [r["label"] for r in records]
    unique_labels, counts = np.unique(labels, return_counts=True)

    summary = {
        "total_images": len(records),
        "class_breakdown": {str(k): int(v) for k, v in zip(unique_labels, counts)},
        "imbalance_ratio": float(max(counts)) / float(min(counts)) if len(counts) > 0 and min(counts) > 0 else 1.0,
        "image_dimensions": {
            "width_min": int(min(widths)),
            "width_max": int(max(widths)),
            "height_min": int(min(heights)),
            "height_max": int(max(heights)),
        },
        "quality": {
            "mean_tissue_ratio": round(float(np.mean(tissue_ratios)), 4),
            "mean_focus_score": round(float(np.mean(focus_scores)), 2),
            "rejected_tiles_count": sum(1 for r in records if not r.get("is_accepted", True)),
        },
    }

    with open(out_dir / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary
