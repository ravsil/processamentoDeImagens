"""Testes das partes novas da entrega final (perturbações oficiais e contrato do CSV)."""

import numpy as np
import pandas as pd

from pathoshift.preprocessing.perturbations import (
    apply_manifest_perturbation,
    parse_parameters,
    random_acquisition_shift,
)
from pathoshift.preprocessing.pipeline import PreprocConfig, Preprocessor


def _img(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(60, 230, size=(64, 64, 3), dtype=np.uint8)


def test_parse_parameters():
    assert parse_parameters("brightness=0.15;contrast=1.20") == {"brightness": 0.15, "contrast": 1.2}


def test_manifest_perturbations_are_deterministic():
    img = _img()
    for kind, params in [("brightness_contrast", "brightness=0.15;contrast=1.20"), ("jpeg", "quality=35"),
                         ("gaussian_blur", "sigma=1.2"), ("gaussian_noise", "stddev=12")]:
        a = apply_manifest_perturbation(img, kind, params, 7)
        b = apply_manifest_perturbation(img, kind, params, 7)
        assert a.dtype == np.uint8 and a.shape == img.shape
        assert np.array_equal(a, b)
        assert not np.array_equal(a, img)


def test_noise_depends_on_seed():
    img = _img()
    a = apply_manifest_perturbation(img, "gaussian_noise", "stddev=12", 1)
    b = apply_manifest_perturbation(img, "gaussian_noise", "stddev=12", 2)
    assert not np.array_equal(a, b)


def test_random_shift_reproducible():
    img = _img()
    a = random_acquisition_shift(img, np.random.default_rng(5))
    b = random_acquisition_shift(img, np.random.default_rng(5))
    assert np.array_equal(a, b)


def test_preprocessor_variants_keep_shape():
    img = _img()
    for cfg in [PreprocConfig(), PreprocConfig(color="reinhard"), PreprocConfig(color="macenko"),
                PreprocConfig(denoise="gaussian"), PreprocConfig(denoise="median", denoise_param=3),
                PreprocConfig(color="reinhard", denoise="gaussian")]:
        out = Preprocessor(cfg)(img)
        assert out.shape == img.shape and out.dtype == np.uint8


def test_prediction_csv_contract(tmp_path):
    from pathoshift.challenge import write_predictions
    prob = np.full((3, 9), 0.01)
    prob[np.arange(3), [0, 4, 8]] = 0.92
    path = tmp_path / "p.csv"
    write_predictions(path, ["a", "b", "c"], prob)
    df = pd.read_csv(path)
    assert list(df.columns) == ["image_id", "predicted_label", "confidence"]
    assert list(df.predicted_label) == ["ADI", "MUC", "TUM"]
    assert df.confidence.between(0, 1).all()


def test_features_finite_on_saturated_image():
    """Imagem toda branca (ex.: brilho/contraste saturando em 255) não pode gerar NaN.

    Regressão: a assimetria (skew) de um canal constante é 0/0; o extrator deve retornar 0.
    """
    from pathoshift.models.classical import HistopathologyFeatureExtractor
    from pathoshift.preprocessing.perturbations import brightness_contrast

    white = np.full((64, 64, 3), 240, dtype=np.uint8)
    saturated = brightness_contrast(white, 0.15, 1.20)   # mesma perturbação de robustness.csv
    assert (saturated == 255).all()

    feats = HistopathologyFeatureExtractor().extract_all(saturated)
    assert np.isfinite(feats).all()
