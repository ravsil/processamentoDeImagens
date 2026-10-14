"""Synthetic histopathology tile generator for local verification and mock pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import cv2
import numpy as np
from PIL import Image


def generate_synthetic_he_tile(
    category: str = "tumor",
    size: Tuple[int, int] = (224, 224),
    seed: int = 42,
) -> np.ndarray:
    """Generate a realistic synthetic H&E tile image."""
    rng = np.random.RandomState(seed)
    h, w = size

    if category == "background":
        # Empty glass slide with faint illumination gradient and slight dust
        bg = rng.normal(245, 4, (h, w, 3)).clip(230, 255).astype(np.uint8)
        return bg

    # Base stroma / cytoplasm: Eosin pink (high R, lower G and B)
    base = np.zeros((h, w, 3), dtype=np.float32)
    base[:, :, 0] = rng.normal(215, 8, (h, w))  # Red
    base[:, :, 1] = rng.normal(130, 10, (h, w)) # Green
    base[:, :, 2] = rng.normal(175, 10, (h, w)) # Blue

    # Stained cell nuclei: Hematoxylin dark purple (darker, blue/violet tone)
    # Tumor tiles: high nuclear density; Normal tiles: lower nuclear density
    n_nuclei = rng.randint(70, 120) if category == "tumor" else rng.randint(15, 35)

    nuclei_color = np.array([100, 45, 120], dtype=np.float32)

    for _ in range(n_nuclei):
        cy = rng.randint(10, h - 10)
        cx = rng.randint(10, w - 10)
        rad_y = rng.randint(4, 9) if category == "tumor" else rng.randint(3, 6)
        rad_x = rng.randint(4, 9) if category == "tumor" else rng.randint(3, 6)
        angle = rng.uniform(0, 180)

        # Draw nucleus ellipse
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (rad_x, rad_y), angle, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (5, 5), 1.5).astype(np.float32) / 255.0

        for c in range(3):
            base[:, :, c] = base[:, :, c] * (1.0 - mask) + nuclei_color[c] * mask

    tile = np.clip(base, 0, 255).astype(np.uint8)
    return tile


def create_synthetic_dataset(
    output_dir: Path | str,
    n_patients: int = 6,
    tiles_per_patient: int = 8,
    classes: Tuple[str, ...] = ("tumor", "normal"),
    size: Tuple[int, int] = (224, 224),
    seed: int = 42,
) -> Path:
    """Create a structured demo histopathology dataset."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)

    for cls in classes:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True, exist_ok=True)

    tile_idx = 0
    for p_id in range(1, n_patients + 1):
        # Assign patient predominantly to one class to test grouping
        cls = classes[p_id % len(classes)]
        cls_dir = root / cls

        for t_id in range(1, tiles_per_patient + 1):
            tile_seed = seed + tile_idx
            tile_img = generate_synthetic_he_tile(category=cls, size=size, seed=tile_seed)

            filename = f"patient_{p_id:03d}_slide_01_tile_{t_id:03d}.png"
            Image.fromarray(tile_img).save(cls_dir / filename)
            tile_idx += 1

    return root
