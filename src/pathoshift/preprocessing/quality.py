"""Quality assessment and background rejection for histopathology tiles."""

from __future__ import annotations

from typing import Dict, Tuple, Any
import cv2
import numpy as np


def compute_tissue_mask(
    image_rgb: np.ndarray,
    saturation_threshold: int = 15,
    max_luminance: int = 245,
) -> Tuple[np.ndarray, float]:
    """Segment tissue area from empty glass background.

    Histopathology tissue (stained with Hematoxylin & Eosin) exhibits higher color
    saturation and lower luminance than empty glass slide background.

    Args:
        image_rgb: RGB image as uint8 numpy array (H, W, 3).
        saturation_threshold: Minimum HSV saturation value [0-255] for tissue.
        max_luminance: Maximum grayscale intensity [0-255] to filter white glass.

    Returns:
        mask: Binary mask (H, W) where 1 indicates tissue.
        tissue_ratio: Fraction of pixels identified as tissue [0.0, 1.0].
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected 3-channel RGB image, got shape {image_rgb.shape}")

    # Convert to HSV for saturation-based tissue segmentation
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    
    # Also check luminance / brightness to exclude pure white glass
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # Tissue condition: sufficient saturation AND not pure white slide background
    tissue_mask = (saturation >= saturation_threshold) & (gray <= max_luminance)
    
    total_pixels = image_rgb.shape[0] * image_rgb.shape[1]
    tissue_ratio = float(np.count_nonzero(tissue_mask)) / float(total_pixels) if total_pixels > 0 else 0.0
    
    return tissue_mask.astype(np.uint8), tissue_ratio


def compute_focus_measure(image_rgb: np.ndarray) -> float:
    """Compute focus/sharpness metric using the variance of the Laplacian.

    Sharp tissue features (cellular membranes, chromatin granules) yield
    higher gradient variations, whereas blurry or out-of-focus tiles yield low variance.

    Args:
        image_rgb: RGB image as uint8 numpy array (H, W, 3).

    Returns:
        float: Laplacian variance score (higher = sharper).
    """
    if image_rgb.ndim == 3:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_rgb
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def evaluate_tile_quality(
    image_rgb: np.ndarray,
    min_tissue_ratio: float = 0.40,
    min_laplacian_var: float = 50.0,
) -> Dict[str, Any]:
    """Evaluate comprehensive tile quality metrics.

    Args:
        image_rgb: RGB image as uint8 numpy array.
        min_tissue_ratio: Minimum ratio of tissue required to accept the tile.
        min_laplacian_var: Minimum focus score required.

    Returns:
        Dict with metrics and overall pass/reject status.
    """
    _, tissue_ratio = compute_tissue_mask(image_rgb)
    focus_score = compute_focus_measure(image_rgb)
    
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    mean_intensity = float(np.mean(gray))
    std_intensity = float(np.std(gray))
    
    is_valid_tissue = tissue_ratio >= min_tissue_ratio
    is_focused = focus_score >= min_laplacian_var
    is_accepted = is_valid_tissue and is_focused
    
    return {
        "tissue_ratio": tissue_ratio,
        "focus_score": focus_score,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "is_valid_tissue": is_valid_tissue,
        "is_focused": is_focused,
        "is_accepted": is_accepted,
    }
