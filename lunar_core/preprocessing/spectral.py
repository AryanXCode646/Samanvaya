"""Hyperspectral continuum extraction for Chandrayaan-2 IIRS."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


class HyperspectralBandSelector:
    """Extract stable continuum and PCA structural bands from IIRS cubes."""

    DEFAULT_MIN_WAVELENGTH_NM = 800.0
    DEFAULT_MAX_WAVELENGTH_NM = 5000.0
    DEFAULT_NUM_BANDS = 256
    CONTINUUM_MIN_NM = 1000.0
    CONTINUUM_MAX_NM = 1250.0

    def __init__(
        self,
        wavelengths: Optional[Sequence[float]] = None,
        num_bands: int = 256,
        min_wavelength_nm: float = 800.0,
        max_wavelength_nm: float = 5000.0,
    ) -> None:
        self.num_bands = num_bands
        if wavelengths is None:
            self.wavelengths = np.linspace(
                min_wavelength_nm, max_wavelength_nm, num_bands, dtype=np.float32
            )
        else:
            values = np.asarray(wavelengths, dtype=np.float32)
            self.wavelengths = values * 1000.0 if np.max(values) < 50.0 else values

    def _standardize_cube_layout(self, cube: np.ndarray) -> np.ndarray:
        arr = np.asarray(cube, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D hyperspectral cube, got shape {arr.shape}")
        bands = len(self.wavelengths)
        if arr.shape[0] == bands:
            return arr
        if arr.shape[2] == bands:
            return np.transpose(arr, (2, 0, 1))
        if arr.shape[1] == bands:
            return np.transpose(arr, (1, 0, 2))
        if arr.shape[0] == arr.shape[1] and arr.shape[2] != arr.shape[0]:
            return np.transpose(arr, (2, 0, 1))
        if arr.shape[2] < arr.shape[0] and arr.shape[2] < arr.shape[1]:
            return np.transpose(arr, (2, 0, 1))
        return arr

    def get_band_indices_for_range(
        self, min_nm: float = 1000.0, max_nm: float = 1250.0
    ) -> np.ndarray:
        indices = np.where((self.wavelengths >= min_nm) & (self.wavelengths <= max_nm))[0]
        if len(indices):
            return indices
        center = 0.5 * (min_nm + max_nm)
        return np.array([int(np.argmin(np.abs(self.wavelengths - center)))], dtype=np.int64)

    def extract_continuum_band(
        self,
        cube: np.ndarray,
        min_nm: float = 1000.0,
        max_nm: float = 1250.0,
        normalize: bool = True,
    ) -> np.ndarray:
        selected = self._standardize_cube_layout(cube)[
            self.get_band_indices_for_range(min_nm, max_nm)
        ]
        continuum = np.nanmean(selected, axis=0)
        if not normalize:
            return continuum.astype(np.float32)
        p1 = float(np.nanpercentile(continuum, 1.0))
        p99 = float(np.nanpercentile(continuum, 99.0))
        normalized = np.clip((continuum - p1) / max(p99 - p1, 1e-5), 0.0, 1.0)
        return np.nan_to_num(normalized, nan=0.5).astype(np.float32)

    def extract_pca_structural_band(
        self, cube: np.ndarray, subsample_ratio: float = 1.0, normalize: bool = True
    ) -> np.ndarray:
        bands, height, width = self._standardize_cube_layout(cube).shape
        flat = self._standardize_cube_layout(cube).reshape(bands, -1).T
        valid = np.all(np.isfinite(flat), axis=1)
        if not np.any(valid):
            return np.zeros((height, width), dtype=np.float32)
        clean = flat[valid]
        mean = np.mean(clean, axis=0, keepdims=True)
        centered = clean - mean
        sample = centered
        if subsample_ratio < 1.0 and len(centered) > 1000:
            sample = centered[:: max(1, int(1.0 / subsample_ratio))]
        _, eigenvectors = np.linalg.eigh(np.cov(sample, rowvar=False))
        projection = np.dot(flat - mean, eigenvectors[:, -1])
        mean_spatial = np.nanmean(self._standardize_cube_layout(cube), axis=0).ravel()
        correlation = np.corrcoef(projection[valid], mean_spatial[valid])[0, 1]
        if correlation < 0:
            projection = -projection
        image = projection.reshape(height, width)
        if not normalize:
            return image.astype(np.float32)
        p1 = float(np.nanpercentile(image, 1.0))
        p99 = float(np.nanpercentile(image, 99.0))
        normalized = np.clip((image - p1) / max(p99 - p1, 1e-5), 0.0, 1.0)
        return np.nan_to_num(normalized, nan=0.5).astype(np.float32)

    def extract_optimal_structural_band(self, cube: np.ndarray, method: str = "continuum") -> np.ndarray:
        if method.lower() == "pca":
            return self.extract_pca_structural_band(cube)
        return self.extract_continuum_band(cube)
