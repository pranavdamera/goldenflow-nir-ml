"""Tests for spectral preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from app.ml.preprocessor import SpectralPreprocessor
from app.utils.synthetic_data import generate_synthetic_spectrum


def test_snv_normalization_has_zero_mean_and_unit_std() -> None:
    """SNV output should be approximately zero-mean and unit-variance."""
    pre = SpectralPreprocessor()
    spectrum = generate_synthetic_spectrum("pure")
    normalized = pre.normalize(spectrum)
    assert float(np.mean(normalized)) == pytest.approx(0.0, abs=1e-5)
    assert float(np.std(normalized)) == pytest.approx(1.0, abs=1e-5)


def test_validate_range_raises_for_out_of_bounds_values() -> None:
    """Range validation should reject values outside 900-1700nm."""
    pre = SpectralPreprocessor()
    bad = np.linspace(800.0, 1800.0, 256, dtype=np.float32)
    with pytest.raises(ValueError):
        pre.validate_range(bad)
