"""Leitura dos manifestos oficiais do pacote PathoShift-CRC.

Somente as imagens listadas em train.csv e validation.csv são utilizadas.
robustness.csv não adiciona imagens: descreve perturbações aplicadas às
imagens de validação (ver pathoshift.preprocessing.perturbations).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from PIL import Image

CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASSES)}


def load_official_manifests(manifest_dir: str | Path) -> Dict[str, pd.DataFrame]:
    """Carrega train/validation/robustness e verifica a integridade das partições."""
    manifest_dir = Path(manifest_dir)
    train = pd.read_csv(manifest_dir / "train.csv", dtype=str)
    val = pd.read_csv(manifest_dir / "validation.csv", dtype=str)
    rob = pd.read_csv(manifest_dir / "robustness.csv", dtype=str)

    # Sanidade: tamanhos oficiais, ausência de duplicatas e de vazamento treino/validação.
    assert len(train) == 9000 and len(val) == 2250 and len(rob) == 2250, "Tamanho inesperado dos manifestos"
    assert train.image_id.is_unique and val.image_id.is_unique and rob.image_id.is_unique
    assert not set(train.image_path) & set(val.image_path), "Vazamento: imagem em treino e validação"
    assert set(rob.source_image_id) == set(val.image_id), "robustness.csv deve referenciar a validação"
    for df in (train, val, rob):
        assert set(df.label) <= set(CLASSES)

    # O rótulo usado na supervisão vem do manifesto (coluna label); nunca do nome do arquivo.
    rob = rob.rename(columns={"source_image_path": "image_path"})
    return {"train": train, "validation": val, "robustness": rob}


def read_tile(data_root: str | Path, rel_path: str) -> np.ndarray:
    """Lê um tile TIFF como RGB uint8 (H, W, 3)."""
    with Image.open(Path(data_root) / rel_path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
