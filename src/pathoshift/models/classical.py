"""Classical machine learning baseline using handcrafted color and texture features."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import joblib
import numpy as np
from scipy import stats
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm


class HistopathologyFeatureExtractor:
    """Extracts color and texture representations from histopathology tile images."""

    def __init__(
        self,
        color_bins: int = 16,
        lbp_points: int = 24,
        lbp_radius: int = 3,
        glcm_distances: Tuple[int, ...] = (1, 3),
        glcm_angles: Tuple[float, ...] = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    ):
        self.color_bins = color_bins
        self.lbp_points = lbp_points
        self.lbp_radius = lbp_radius
        self.glcm_distances = glcm_distances
        self.glcm_angles = glcm_angles

    def extract_color_features(self, image_rgb: np.ndarray) -> np.ndarray:
        """Extract color channel histograms and statistical moments (RGB + HSV)."""
        feats = []

        # RGB and HSV spaces
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

        for space_img in [image_rgb, hsv]:
            for c in range(3):
                channel = space_img[:, :, c]
                # Moments: mean, std, skewness
                feats.append(float(np.mean(channel)))
                feats.append(float(np.std(channel)))
                feats.append(float(stats.skew(channel.ravel())))

                # Histogram
                hist, _ = np.histogram(channel, bins=self.color_bins, range=(0, 256), density=True)
                feats.extend(hist.tolist())

        return np.array(feats, dtype=np.float32)

    def extract_texture_features(self, image_rgb: np.ndarray) -> np.ndarray:
        """Extract GLCM properties and Local Binary Pattern (LBP) histogram."""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        feats = []

        # 1. GLCM features
        # Rescale grayscale to 64 levels for robust, fast GLCM computation
        gray_scaled = (gray // 4).astype(np.uint8)
        glcm = graycomatrix(
            gray_scaled,
            distances=self.glcm_distances,
            angles=self.glcm_angles,
            levels=64,
            symmetric=True,
            normed=True,
        )

        for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
            values = graycoprops(glcm, prop).ravel()
            feats.append(float(np.mean(values)))
            feats.append(float(np.std(values)))

        # 2. Local Binary Patterns (uniform)
        lbp = local_binary_pattern(
            gray, P=self.lbp_points, R=self.lbp_radius, method="uniform"
        )
        n_bins = self.lbp_points + 2
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        feats.extend(lbp_hist.tolist())

        return np.array(feats, dtype=np.float32)

    def extract_all(self, image_rgb: np.ndarray) -> np.ndarray:
        """Concatenate color and texture features into a single feature vector."""
        color_feats = self.extract_color_features(image_rgb)
        texture_feats = self.extract_texture_features(image_rgb)
        return np.concatenate([color_feats, texture_feats])


class ClassicalBaselineModel:
    """Classical classifier (Random Forest or SVM) with feature scaling."""

    def __init__(self, classifier_type: str = "rf", n_estimators: int = 150, random_state: int = 42):
        self.classifier_type = classifier_type.lower()
        self.extractor = HistopathologyFeatureExtractor()
        self.scaler = StandardScaler()
        self.random_state = random_state

        if self.classifier_type == "rf":
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced",
            )
        elif self.classifier_type == "svm":
            self.model = SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=random_state,
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassicalBaselineModel":
        """Fit scaler and classifier."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def save(self, filepath: str | Path) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str | Path) -> "ClassicalBaselineModel":
        return joblib.load(filepath)
