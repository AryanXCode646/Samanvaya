"""Hyperspectral continuum extraction for Chandrayaan-2 IIRS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np


@dataclass
class SpectralCube:
    """Bands-first spectral cube with deterministic 2-D representations."""

    data: np.ndarray
    fill_value: Optional[float] = None
    wavelengths: Optional[np.ndarray] = None
    wavelength_units: Optional[str] = None
    wavelength_source: str = "WAVELENGTH_METADATA_UNAVAILABLE"
    axis_order: str = "bands,height,width"
    metadata_provenance: Optional[str] = None

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise ValueError(f"SpectralCube expects (bands, height, width), got {self.data.shape}")

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.data.shape)

    def to_image(self, method: str = "band_mean", components: int = 1) -> np.ndarray:
        selector = HyperspectralBandSelector(
            wavelengths=self.wavelengths,
            num_bands=self.shape[0],
        )
        if method == "band_mean":
            image = np.nanmean(np.asarray(self.data, dtype=np.float32), axis=0)
            return np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if method == "pca":
            if components != 1:
                raise ValueError("The registration representation currently supports exactly one PCA component")
            return selector.extract_pca_structural_band(np.asarray(self.data), normalize=False)
        raise ValueError(f"Unsupported spectral representation: {method}")

    def representation_provenance(self, method: str, components: int = 1) -> dict[str, Any]:
        """Describe exactly how a deterministic 2-D registration image was produced."""
        return {
            "input": "IIRS spectral cube",
            "axis_order": self.axis_order,
            "bands": self.shape[0],
            "wavelength_units": self.wavelength_units,
            "wavelength_source": self.wavelength_source,
            "wavelength_range": (
                [float(np.nanmin(self.wavelengths)), float(np.nanmax(self.wavelengths))]
                if self.wavelengths is not None and len(self.wavelengths)
                else None
            ),
            "representation": method,
            "components": components if method == "pca" else None,
            "metadata_provenance": self.metadata_provenance,
        }


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
            # Synthetic/unit-test selector compatibility; real SpectralCube provenance
            # remains explicitly unavailable unless authoritative wavelengths are passed.
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
        bands = len(self.wavelengths) if self.wavelengths is not None else self.num_bands
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

    def filter_bad_bands_and_pixels(
        self,
        cube: np.ndarray,
        max_nan_ratio: float = 0.40,
        min_variance: float = 1e-7,
    ) -> tuple[np.ndarray, list[int], dict[str, Any]]:
        """Filter out dead, saturated, or high-noise hyperspectral channels and clean sparse invalid pixels."""
        arr = self._standardize_cube_layout(cube)
        bands, height, width = arr.shape
        total_pixels = height * width

        kept_indices: list[int] = []
        cleaned_bands: list[np.ndarray] = []

        for b in range(bands):
            band_slice = arr[b]
            nan_count = int(np.count_nonzero(~np.isfinite(band_slice)))
            if nan_count / max(total_pixels, 1) > max_nan_ratio:
                continue

            valid_mask = np.isfinite(band_slice)
            if not np.any(valid_mask):
                continue

            var = float(np.var(band_slice[valid_mask]))
            if var < min_variance:
                continue

            # Fill remaining sparse non-finite pixels with median of valid regolith pixels
            median_val = float(np.median(band_slice[valid_mask]))
            clean_slice = np.where(valid_mask, band_slice, median_val).astype(np.float32)
            cleaned_bands.append(clean_slice)
            kept_indices.append(b)

        if not cleaned_bands:
            # Fallback to zero-filled band if all bands were degenerate
            cleaned_cube = np.zeros((1, height, width), dtype=np.float32)
            kept_indices = [0]
        else:
            cleaned_cube = np.stack(cleaned_bands, axis=0)

        metadata = {
            "total_bands_input": bands,
            "bands_retained": len(kept_indices),
            "retained_indices": kept_indices,
            "discarded_band_count": bands - len(kept_indices),
            "max_nan_threshold": max_nan_ratio,
            "min_variance_threshold": min_variance,
        }
        return cleaned_cube, kept_indices, metadata

    def get_band_indices_for_range(
        self, min_nm: float = 1000.0, max_nm: float = 1250.0
    ) -> np.ndarray:
        if self.wavelengths is None:
            raise ValueError("WAVELENGTH_METADATA_UNAVAILABLE: cannot select a wavelength range")
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
        clean_cube, kept, _ = self.filter_bad_bands_and_pixels(cube)
        if self.wavelengths is not None and len(self.wavelengths) == self._standardize_cube_layout(cube).shape[0]:
            sub_wavelengths = np.asarray(self.wavelengths)[kept]
            indices = np.where((sub_wavelengths >= min_nm) & (sub_wavelengths <= max_nm))[0]
            if len(indices) == 0:
                indices = np.array([int(np.argmin(np.abs(sub_wavelengths - 0.5 * (min_nm + max_nm))))])
            selected = clean_cube[indices]
        else:
            selected = clean_cube
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
        clean_cube, kept, _ = self.filter_bad_bands_and_pixels(cube)
        bands, height, width = clean_cube.shape
        if bands == 1:
            img = clean_cube[0]
            if not normalize:
                return img.astype(np.float32)
            p1, p99 = float(np.nanpercentile(img, 1.0)), float(np.nanpercentile(img, 99.0))
            return np.nan_to_num(np.clip((img - p1) / max(p99 - p1, 1e-5), 0.0, 1.0), nan=0.5).astype(np.float32)

        flat = clean_cube.reshape(bands, -1).T
        valid = np.all(np.isfinite(flat), axis=1)
        if not np.any(valid):
            return np.zeros((height, width), dtype=np.float32)
        clean = flat[valid]
        mean = np.mean(clean, axis=0, keepdims=True)
        centered = clean - mean
        sample = centered
        if subsample_ratio < 1.0 and len(centered) > 1000:
            sample = centered[:: max(1, int(1.0 / subsample_ratio))]
        cov = np.cov(sample, rowvar=False)
        if cov.ndim == 0 or cov.size == 1:
            eigenvectors = np.ones((1, 1), dtype=np.float32)
        else:
            _, eigenvectors = np.linalg.eigh(cov)
        projection = np.dot(flat - mean, eigenvectors[:, -1])
        mean_spatial = np.nanmean(clean_cube, axis=0).ravel()
        correlation = np.corrcoef(projection[valid], mean_spatial[valid])[0, 1] if len(clean) > 2 else 1.0
        if not np.isnan(correlation) and correlation < 0:
            projection = -projection
        image = projection.reshape(height, width)
        if not normalize:
            return image.astype(np.float32)
        p1 = float(np.nanpercentile(image, 1.0))
        p99 = float(np.nanpercentile(image, 99.0))
        normalized = np.clip((image - p1) / max(p99 - p1, 1e-5), 0.0, 1.0)
        return np.nan_to_num(normalized, nan=0.5).astype(np.float32)

    def extract_2d_registration_representation(
        self,
        cube: np.ndarray,
        method: str = "auto",
        normalize: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Extract a scientifically defensible 2-D registration image with provenance matching PS 26166."""
        clean_cube, kept_indices, filter_meta = self.filter_bad_bands_and_pixels(cube)
        can_do_continuum = (
            self.wavelengths is not None
            and len(self.wavelengths) == self._standardize_cube_layout(cube).shape[0]
        )

        if method == "continuum" or (method == "auto" and can_do_continuum):
            image = self.extract_continuum_band(clean_cube, normalize=normalize)
            prov = {
                "representation_type": "continuum_band_mean",
                "bands_used": kept_indices,
                "dimensionality_reduction": "continuum_window_average",
                "spectral_preprocessing": "bad_pixel_and_dead_band_filtering",
                "wavelength_range_nm": [1000.0, 1250.0],
                "filter_audit": filter_meta,
            }
        else:
            image = self.extract_pca_structural_band(clean_cube, normalize=normalize)
            prov = {
                "representation_type": "robust_pca_structural_band",
                "bands_used": kept_indices,
                "dimensionality_reduction": "PCA_first_principal_component",
                "spectral_preprocessing": "bad_pixel_and_dead_band_filtering",
                "filter_audit": filter_meta,
            }

        return image, prov

    def extract_optimal_structural_band(self, cube: np.ndarray, method: str = "continuum") -> np.ndarray:
        if method.lower() == "pca":
            return self.extract_pca_structural_band(cube)
        return self.extract_continuum_band(cube)

