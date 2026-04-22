"""Spectral preprocessing utilities for NIR inference."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


class SpectralPreprocessor:
    """Preprocess raw NIR arrays into model-ready tensors.

    Pipeline order: validate → smooth → normalize.

    Savitzky-Golay smoothing runs *before* SNV normalization deliberately:
    high-frequency noise inflates a spectrum's variance estimate, which
    would corrupt the mean/std calculation SNV depends on. Smoothing first
    gives a cleaner baseline for normalization.

    SNV (Standard Normal Variate) removes multiplicative scatter effects —
    baseline shifts caused by differences in particle size, path length,
    or detector-sample distance across measurements. Because each spectrum
    is normalized by its own statistics, SNV is insensitive to inter-scan
    intensity drift without needing an external reference.
    """

    def validate_range(self, array: np.ndarray) -> None:
        """Validate that spectral points lie inside the 900-1700nm range.

        Args:
            array: A 1D spectral array with 256 points.

        Raises:
            ValueError: If the array length is not 256 or values are out of range.
        """
        if array.ndim != 1 or array.shape[0] != 256:
            raise ValueError("Input spectral array must be one-dimensional with 256 values.")
        if np.any(array < 900.0) or np.any(array > 1700.0):
            raise ValueError("Spectral array contains values outside the 900-1700nm range.")

    def normalize(self, array: np.ndarray) -> np.ndarray:
        """Apply Standard Normal Variate normalization to one spectrum."""
        mean = float(np.mean(array))
        std = float(np.std(array))
        if std == 0.0:
            raise ValueError("Cannot normalize spectrum with zero standard deviation.")
        return (array - mean) / std

    def smooth(self, array: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay smoothing (window=11, polyorder=2)."""
        return savgol_filter(array, window_length=11, polyorder=2, mode="interp")

    def full_pipeline(self, raw_array: np.ndarray) -> np.ndarray:
        """Run validation, smoothing, and SNV normalization in order."""
        array = np.asarray(raw_array, dtype=np.float32)
        self.validate_range(array)
        smoothed = self.smooth(array)
        normalized = self.normalize(smoothed)
        return normalized.astype(np.float32)
