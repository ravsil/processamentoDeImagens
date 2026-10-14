"""Visualization and plotting package."""

from pathoshift.visualization.eda import (
    plot_class_distribution,
    plot_color_channels,
    plot_quality_metrics,
    run_full_eda,
)
from pathoshift.visualization.report_plots import (
    plot_confusion_matrix,
    plot_error_gallery,
    plot_training_curves,
)

__all__ = [
    "plot_class_distribution",
    "plot_color_channels",
    "plot_quality_metrics",
    "run_full_eda",
    "plot_confusion_matrix",
    "plot_error_gallery",
    "plot_training_curves",
]
