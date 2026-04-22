"""Synthetic spectral generator for local testing and demos."""

from __future__ import annotations

from typing import Dict

import numpy as np

# SYNTHETIC DATA ONLY. Real HoneyPrint spectral database is proprietary.


def generate_synthetic_spectrum(label: str = "pure", noise_level: float = 0.02) -> np.ndarray:
    """Generate a realistic-looking synthetic 256-point NIR spectrum.

    The generated values are represented on the 900-1700nm domain and
    intentionally vary by label class using distinct sine-wave mixtures.

    Args:
        label: Class profile to simulate.
        noise_level: Gaussian noise scale added to base spectral signature.

    Returns:
        A float32 numpy array of shape (256,) with values in [900, 1700].
    """
    x = np.linspace(0.0, 2.0 * np.pi, 256, dtype=np.float32)
    base = 1300.0 + 40.0 * np.sin(1.5 * x) + 25.0 * np.sin(3.1 * x + 0.5)

    profiles: Dict[str, np.ndarray] = {
        "pure": base + 8.0 * np.sin(5.0 * x),
        "rice_syrup": base + 18.0 * np.sin(2.2 * x + 1.1),
        "hfcs": base + 22.0 * np.sin(2.9 * x + 0.7),
        "jaggery_syrup": base + 20.0 * np.cos(2.1 * x + 0.3),
        "invert_sugar": base + 15.0 * np.sin(4.2 * x + 0.9),
        "sugar_fed": base + 12.0 * np.cos(4.7 * x + 0.4),
        "unknown": base + 10.0 * np.sin(6.5 * x + 0.2),
    }
    spectrum = profiles.get(label, profiles["unknown"]).copy()
    spectrum += np.random.normal(0.0, noise_level * 50.0, size=256).astype(np.float32)
    return np.clip(spectrum, 900.0, 1700.0).astype(np.float32)
