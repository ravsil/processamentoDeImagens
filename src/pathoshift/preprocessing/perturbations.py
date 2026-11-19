"""Perturbações de robustez.

1. apply_manifest_perturbation: aplica, de forma determinística, a perturbação
   descrita em cada linha de robustness.csv (tipo, parâmetros e seed).
2. random_acquisition_shift: gerador aleatório de mudanças de aquisição usado
   (a) como aumentação de treino e (b) para criar uma validação perturbada
   "proxy" usada na seleção de configurações. Ele NÃO usa robustness.csv:
   as faixas de parâmetros são fixas e definidas a priori (ver README).
"""

from __future__ import annotations

import io
from typing import Dict

import cv2
import numpy as np
from PIL import Image


def parse_parameters(text: str) -> Dict[str, float]:
    """'brightness=0.15;contrast=1.20' -> {'brightness': 0.15, 'contrast': 1.2}"""
    out: Dict[str, float] = {}
    for part in str(text).split(";"):
        if part.strip():
            k, v = part.split("=")
            out[k.strip()] = float(v)
    return out


# ---------------------------------------------------------------- primitivas
BRIGHTNESS_CONTRAST_FORMULAS = ("A", "B", "C", "D")


def brightness_contrast(img: np.ndarray, brightness: float, contrast: float, formula: str = "A") -> np.ndarray:
    """Aplica variação de brilho e contraste segundo uma das convenções possíveis.

    Fórmulas (x em [0, 255]):
      A: y = c * x + 255 * b                        (OpenCV / Albumentations alpha/beta — padrão)
      B: y = (x - 128) * c + 128 + 255 * b          (contraste centrado no cinza médio 128)
      C: y = (x - mu) * c + mu + 255 * b            (contraste centrado na média da imagem)
      D: x1 = clip(x * (1 + b)); y = (x1 - mu1) * c + mu1  (brilho multiplicativo + contraste estilo PIL/torchvision)
    """
    x = img.astype(np.float32)
    b = float(brightness)
    c = float(contrast)
    if formula == "A":
        y = x * c + b * 255.0
    elif formula == "B":
        y = (x - 128.0) * c + 128.0 + b * 255.0
    elif formula == "C":
        mu = float(x.mean())
        y = (x - mu) * c + mu + b * 255.0
    elif formula == "D":
        x1 = np.clip(np.rint(x * (1.0 + b)), 0, 255).astype(np.uint8)
        mu1 = float(cv2.cvtColor(x1, cv2.COLOR_RGB2GRAY).mean())
        y = (x1.astype(np.float32) - mu1) * c + mu1
    else:
        raise ValueError(f"Fórmula de brightness_contrast desconhecida: {formula}")
    return np.clip(np.rint(y), 0, 255).astype(np.uint8)


def jpeg_compress(img: np.ndarray, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    with Image.open(buf) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma),
                            borderType=cv2.BORDER_REFLECT_101)


def gaussian_noise(img: np.ndarray, stddev: float, rng: np.random.Generator) -> np.ndarray:
    y = img.astype(np.float32) + rng.normal(0.0, float(stddev), size=img.shape).astype(np.float32)
    return np.clip(np.rint(y), 0, 255).astype(np.uint8)


def hsv_jitter(img: np.ndarray, hue_shift: float, sat_scale: float) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


# ------------------------------------------------------- manifesto oficial
def apply_manifest_perturbation(
    img: np.ndarray,
    perturbation: str,
    parameters: str,
    seed: int,
    bc_formula: str = "A",
) -> np.ndarray:
    """Aplica a perturbação de uma linha de robustness.csv (determinística)."""
    p = parse_parameters(parameters)
    rng = np.random.default_rng(int(seed))
    if perturbation == "brightness_contrast":
        return brightness_contrast(img, p["brightness"], p["contrast"], formula=bc_formula)
    if perturbation == "jpeg":
        return jpeg_compress(img, int(p["quality"]))
    if perturbation == "gaussian_blur":
        return gaussian_blur(img, p["sigma"])
    if perturbation == "gaussian_noise":
        return gaussian_noise(img, p["stddev"], rng)
    raise ValueError(f"Perturbação desconhecida: {perturbation}")


# ------------------------------------------- mudanças aleatórias (aumentação)
SHIFT_KINDS = ("color", "jpeg", "blur", "noise")


def random_acquisition_shift(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sorteia UMA mudança de aquisição genérica com faixas fixas definidas a priori.

    Faixas inspiradas em aumentações usuais para H&E (Tellez et al., 2019):
      color : brilho U(-0.2, 0.2), contraste U(0.75, 1.3), matiz U(-6, 6), saturação U(0.8, 1.2)
      jpeg  : qualidade U{20..90}
      blur  : sigma U(0.4, 2.0)
      noise : desvio U(3, 20)
    """
    kind = SHIFT_KINDS[int(rng.integers(len(SHIFT_KINDS)))]
    if kind == "color":
        out = hsv_jitter(img, rng.uniform(-6, 6), rng.uniform(0.8, 1.2))
        return brightness_contrast(out, rng.uniform(-0.2, 0.2), rng.uniform(0.75, 1.3))
    if kind == "jpeg":
        return jpeg_compress(img, int(rng.integers(20, 91)))
    if kind == "blur":
        return gaussian_blur(img, rng.uniform(0.4, 2.0))
    return gaussian_noise(img, rng.uniform(3, 20), rng)
