"""Unit tests for preprocessing, stain normalization, and quality metrics."""

import numpy as np
import pytest

from pathoshift.preprocessing.quality import (
    compute_focus_measure,
    compute_tissue_mask,
    evaluate_tile_quality,
)
from pathoshift.preprocessing.stain_norm import (
    MacenkoNormalizer,
    ReinhardNormalizer,
)


def test_tissue_mask_detection():
    # Synthetic white background (glass slide)
    white_bg = np.full((100, 100, 3), 250, dtype=np.uint8)
    mask_bg, ratio_bg = compute_tissue_mask(white_bg)
    assert ratio_bg == 0.0
    assert np.all(mask_bg == 0)

    # Synthetic stained tissue (pink/purple H&E-like region)
    tissue = np.full((100, 100, 3), [180, 80, 140], dtype=np.uint8)
    mask_tissue, ratio_tissue = compute_tissue_mask(tissue)
    assert ratio_tissue > 0.95
    assert np.all(mask_tissue == 1)


def test_focus_measure():
    # Flat image (zero variance / completely blurry)
    flat = np.full((100, 100, 3), 128, dtype=np.uint8)
    score_flat = compute_focus_measure(flat)
    assert score_flat == 0.0

    # Textured image (high sharp variance)
    np.random.seed(42)
    textured = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    score_textured = compute_focus_measure(textured)
    assert score_textured > 100.0


def test_reinhard_normalization():
    source = np.full((50, 50, 3), [200, 100, 150], dtype=np.uint8)
    normalizer = ReinhardNormalizer()
    norm = normalizer.transform(source)

    assert norm.shape == (50, 50, 3)
    assert norm.dtype == np.uint8


def test_macenko_normalization():
    # Create realistic synthetic H&E tile
    rng = np.random.RandomState(42)
    base = np.full((100, 100, 3), [190, 80, 140], dtype=np.uint8)
    noise = rng.randint(-15, 15, (100, 100, 3)).astype(np.int16)
    tile = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    normalizer = MacenkoNormalizer()
    norm = normalizer.transform(tile)

    assert norm.shape == (100, 100, 3)
    assert norm.dtype == np.uint8
