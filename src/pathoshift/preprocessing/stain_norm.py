"""Stain normalization algorithms for histopathology images (Macenko and Reinhard methods)."""

from __future__ import annotations

from typing import Optional, Tuple
import cv2
import numpy as np


class MacenkoNormalizer:
    """Macenko stain normalizer using optical density decomposition and SVD.

    Reference:
        Macenko, M., et al. "A method for normalizing histology slides for
        quantitative analysis." ISBI 2009.
    """

    # Standard default H&E reference vectors and maximum concentrations
    DEFAULT_HERef = np.array([
        [0.5626, 0.2159],
        [0.7201, 0.8012],
        [0.4062, 0.5581],
    ], dtype=np.float64)

    DEFAULT_maxCRef = np.array([1.9705, 1.0308], dtype=np.float64)

    def __init__(self, beta: float = 0.15, alpha: float = 1.0):
        """
        Args:
            beta: Optical density threshold for transparent / background removal.
            alpha: Percentile threshold (e.g. 1.0 -> 1st and 99th percentiles).
        """
        self.beta = beta
        self.alpha = alpha
        self.HERef = self.DEFAULT_HERef.copy()
        self.maxCRef = self.DEFAULT_maxCRef.copy()

    @staticmethod
    def _rgb_to_od(image_rgb: np.ndarray) -> np.ndarray:
        """Convert uint8 RGB image to Optical Density space."""
        img = image_rgb.astype(np.float64)
        # Avoid log(0)
        img = np.maximum(img, 1.0)
        return -np.log10(img / 255.0)

    @staticmethod
    def _od_to_rgb(od: np.ndarray) -> np.ndarray:
        """Convert Optical Density to uint8 RGB image."""
        rgb = 255.0 * np.power(10.0, -od)
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def fit(self, target_rgb: np.ndarray) -> "MacenkoNormalizer":
        """Fit reference stain vectors and max concentration from a target image."""
        stain_matrix, max_c = self._get_stain_matrix(target_rgb)
        if stain_matrix is not None and max_c is not None:
            self.HERef = stain_matrix
            self.maxCRef = max_c
        return self

    def _get_stain_matrix(
        self, image_rgb: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract H&E stain vectors and max concentration using SVD."""
        try:
            od = self._rgb_to_od(image_rgb).reshape(-1, 3)
            # Filter background pixels
            mask = np.all(od > self.beta, axis=1)
            od_tissue = od[mask]

            if len(od_tissue) < 100:
                return None, None

            # SVD on covariance
            _, _, vh = np.linalg.svd(od_tissue, full_matrices=False)
            # Plane spanned by the two largest singular vectors
            plane = vh[:2, :]

            # Project OD onto the plane
            projected = np.dot(od_tissue, plane.T)

            # Calculate angles
            angles = np.arctan2(projected[:, 1], projected[:, 0])

            # Robust min and max percentiles
            min_phi = np.percentile(angles, self.alpha)
            max_phi = np.percentile(angles, 100.0 - self.alpha)

            # Reconstruct extremal stain vectors
            v_min = np.dot(plane.T, np.array([np.cos(min_phi), np.sin(min_phi)]))
            v_max = np.dot(plane.T, np.array([np.cos(max_phi), np.sin(max_phi)]))

            # Ensure 1st vector is Hematoxylin (higher absorption in red than eosin)
            if v_min[0] > v_max[0]:
                stain_matrix = np.stack([v_min, v_max], axis=1)
            else:
                stain_matrix = np.stack([v_max, v_min], axis=1)

            # Normalize column vectors to unit length
            stain_matrix = stain_matrix / np.linalg.norm(stain_matrix, axis=0, keepdims=True)

            # Compute concentration maps for the tissue
            c, _, _, _ = np.linalg.lstsq(stain_matrix, od_tissue.T, rcond=None)
            max_c = np.percentile(c, 99.0, axis=1)

            return stain_matrix, max_c
        except Exception:
            return None, None

    def transform(self, image_rgb: np.ndarray) -> np.ndarray:
        """Apply Macenko normalization to an RGB image."""
        try:
            h, w, c_dim = image_rgb.shape
            stain_matrix, max_c = self._get_stain_matrix(image_rgb)
            if stain_matrix is None or max_c is None:
                return image_rgb

            od = self._rgb_to_od(image_rgb).reshape(-1, 3)
            # Decompose current image into concentrations: OD = C * S^T -> C = OD * pinv(S^T)
            concentrations, _, _, _ = np.linalg.lstsq(stain_matrix, od.T, rcond=None)

            # Normalize concentrations to target reference
            concentrations[0, :] *= (self.maxCRef[0] / max(max_c[0], 1e-6))
            concentrations[1, :] *= (self.maxCRef[1] / max(max_c[1], 1e-6))

            # Reconstruct normalized optical density using reference vectors
            od_norm = np.dot(self.HERef, concentrations).T
            rgb_norm = self._od_to_rgb(od_norm).reshape(h, w, c_dim)
            return rgb_norm
        except Exception:
            # Fallback to source image if numerical error occurs
            return image_rgb


class ReinhardNormalizer:
    """Reinhard color transfer normalizer in Lab color space.

    Reference:
        Reinhard, E., et al. "Color transfer between images." IEEE CGA 2001.
    """

    def __init__(self):
        # Default target Lab stats (typical H&E appearance)
        self.target_mean = np.array([148.0, 147.0, 108.0], dtype=np.float32)
        self.target_std = np.array([32.0, 18.0, 14.0], dtype=np.float32)

    def fit(self, target_rgb: np.ndarray) -> "ReinhardNormalizer":
        """Compute target channel means and standard deviations in Lab color space."""
        lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        self.target_mean = np.mean(lab, axis=(0, 1))
        self.target_std = np.std(lab, axis=(0, 1)) + 1e-6
        return self

    def transform(self, image_rgb: np.ndarray) -> np.ndarray:
        """Apply Reinhard color transfer to input RGB image."""
        try:
            lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
            src_mean = np.mean(lab, axis=(0, 1))
            src_std = np.std(lab, axis=(0, 1)) + 1e-6

            # Transfer color distribution
            norm_lab = (lab - src_mean) * (self.target_std / src_std) + self.target_mean
            norm_lab = np.clip(norm_lab, 0, 255).astype(np.uint8)
            norm_rgb = cv2.cvtColor(norm_lab, cv2.COLOR_LAB2RGB)
            return norm_rgb
        except Exception:
            return image_rgb
