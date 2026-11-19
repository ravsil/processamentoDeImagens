"""Pré-processamento configurável aplicado antes da extração de atributos.

Ordem: (1) suavização/redução de ruído  ->  (2) normalização de cor.
A suavização vem primeiro porque ruído e artefatos de JPEG distorcem as
estatísticas usadas na estimativa de cor (médias/desvios em LAB ou vetores
de corante em densidade óptica).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import cv2
import numpy as np

from pathoshift.preprocessing.stain_norm import MacenkoNormalizer, ReinhardNormalizer


@dataclass(frozen=True)
class PreprocConfig:
    color: str = "none"          # none | reinhard | macenko
    denoise: str = "none"        # none | gaussian | median
    denoise_param: float = 1.0   # sigma (gaussian) ou tamanho do kernel (median)

    def to_dict(self):
        return asdict(self)


class Preprocessor:
    def __init__(self, cfg: PreprocConfig, reinhard_target: Optional[dict] = None):
        self.cfg = cfg
        self.reinhard = ReinhardNormalizer()
        if reinhard_target is not None:
            self.reinhard.target_mean = np.asarray(reinhard_target["mean"], dtype=np.float32)
            self.reinhard.target_std = np.asarray(reinhard_target["std"], dtype=np.float32)
        self.macenko = MacenkoNormalizer()

    def __call__(self, img: np.ndarray) -> np.ndarray:
        c = self.cfg
        if c.denoise == "gaussian":
            img = cv2.GaussianBlur(img, (0, 0), sigmaX=c.denoise_param)
        elif c.denoise == "median":
            img = cv2.medianBlur(img, int(c.denoise_param))
        elif c.denoise != "none":
            raise ValueError(c.denoise)

        if c.color == "reinhard":
            img = self.reinhard.transform(img)
        elif c.color == "macenko":
            img = self.macenko.transform(img)
        elif c.color != "none":
            raise ValueError(c.color)
        return img
