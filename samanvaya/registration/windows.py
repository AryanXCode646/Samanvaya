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


def extract_registration_windows(source_product: Any, reference_product: Any) -> RegistrationWindows:
    """Read only the selected registration windows and preserve original offsets.

    Without authoritative comparable footprints, the safe result is the full native
    raster window with `OVERLAP_UNKNOWN`, never a claimed physical overlap.
    """
    with rasterio.open(Path(source_product.image_path)) as source, rasterio.open(Path(reference_product.image_path)) as reference:
        source_window = Window(0, 0, source.width, source.height)
        reference_window = Window(0, 0, reference.width, reference.height)
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
        )
