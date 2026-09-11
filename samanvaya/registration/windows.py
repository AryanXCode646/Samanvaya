"""Windowed raster extraction for real registration inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window

from samanvaya.registration.coordinates import CoordinateAudit


@dataclass
class RegistrationWindows:
    source_window: Window
    reference_window: Window
    source_transform: Any
    reference_transform: Any
    source_valid_mask: np.ndarray
    reference_valid_mask: np.ndarray
    source_image: np.ndarray
    reference_image: np.ndarray
    scale_ratio: float | None
    overlap_status: str
    geometry_method: str
    source_coordinates: CoordinateAudit
    reference_coordinates: CoordinateAudit
    spectral_representation: str = "panchromatic_direct"
    spectral_provenance: dict[str, Any] | None = None


def extract_registration_windows(source_product: Any, reference_product: Any) -> RegistrationWindows:
    """Read only the selected registration windows and preserve original offsets.

    Without authoritative comparable footprints, the safe result is the full native
    raster window with `OVERLAP_UNKNOWN`, never a claimed physical overlap.
    """
    with rasterio.open(Path(source_product.image_path)) as source, rasterio.open(Path(reference_product.image_path)) as reference:
        source_window = Window(0, 0, source.width, source.height)
        reference_window = Window(0, 0, reference.width, reference.height)

        is_iirs = (
            str(getattr(source_product, "instrument", "") or "").upper() == "IIRS"
            or "IIR" in str(getattr(source_product, "product_id", "") or "").upper()
            or source.count > 1
        )
        spectral_representation = "panchromatic_direct"
        spectral_provenance = None

        if is_iirs:
            wavelengths = getattr(source_product, "wavelengths", None)
            if source.count > 1:
                if wavelengths is not None and len(wavelengths) == source.count:
                    from lunar_core.preprocessing.spectral import HyperspectralBandSelector
                    cube_data = source.read(window=source_window, masked=True).astype(np.float32).filled(np.nan)
                    selector = HyperspectralBandSelector(wavelengths=wavelengths, num_bands=source.count)
                    source_image = selector.extract_continuum_band(cube_data, normalize=True)
                    spectral_representation = "wavelength_continuum_band"
                    spectral_provenance = {
                        "instrument": "IIRS",
                        "band_count": source.count,
                        "wavelength_min": float(np.min(wavelengths)),
                        "wavelength_max": float(np.max(wavelengths)),
                        "method": "continuum_1000_1250nm",
                    }
                else:
                    raise ValueError(
                        "IIRS_REPRESENTATION_UNCERTAIN: Authoritative spectral band calibration/wavelength metadata is unavailable to justify a 2-D registration continuum."
                    )
            elif str(getattr(source_product, "instrument", "") or "").upper() == "IIRS":
                raise ValueError(
                    "IIRS_REPRESENTATION_UNCERTAIN: Single-band raster labeled IIRS lacks spectral continuum metadata."
                )
            else:
                source_image = source.read(1, window=source_window, masked=True).astype(np.float32).filled(np.nan)
        else:
            source_image = source.read(1, window=source_window, masked=True).astype(np.float32).filled(np.nan)

        reference_image = reference.read(1, window=reference_window, masked=True).astype(np.float32).filled(np.nan)
        source_mask = np.isfinite(source_image)
        reference_mask = np.isfinite(reference_image)
        ratio = None
        if source_product.gsd_m and reference_product.gsd_m and source_product.gsd_m > 0 and reference_product.gsd_m > 0:
            ratio = max(source_product.gsd_m, reference_product.gsd_m) / min(source_product.gsd_m, reference_product.gsd_m)
        return RegistrationWindows(
            source_window=source_window,
            reference_window=reference_window,
            source_transform=source.window_transform(source_window),
            reference_transform=reference.window_transform(reference_window),
            source_valid_mask=source_mask,
            reference_valid_mask=reference_mask,
            source_image=source_image,
            reference_image=reference_image,
            scale_ratio=ratio,
            overlap_status="OVERLAP_UNKNOWN",
            geometry_method="native_full_window_no_authoritative_pixel_overlap",
            source_coordinates=CoordinateAudit(source_window.col_off, source_window.row_off),
            reference_coordinates=CoordinateAudit(reference_window.col_off, reference_window.row_off),
            spectral_representation=spectral_representation,
            spectral_provenance=spectral_provenance,
        )
