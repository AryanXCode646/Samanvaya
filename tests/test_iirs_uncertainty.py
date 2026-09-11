"""
Test IIRS representation uncertainty guard (SIH PS 26166 Phase 8).
Verifies that arbitrary multi-band IIRS cubes without wavelength calibration
raise IIRS_REPRESENTATION_UNCERTAIN and are not blindly collapsed to band 0.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from lunar_core.data_io.mission_product import MissionProduct
from samanvaya.registration.windows import extract_registration_windows
from samanvaya.validation.real_registration import register_products


def test_iirs_cube_without_wavelengths_triggers_uncertainty(tmp_path: Path):
    source_raster = tmp_path / "ch2_iirs_test.tif"
    ref_raster = tmp_path / "ref.tif"
    transform = from_origin(0.0, 100.0, 1.0, 1.0)

    # Multi-band cube (4 bands)
    cube_data = np.ones((4, 64, 64), dtype=np.float32) * 100.0
    with rasterio.open(
        source_raster, "w", driver="GTiff", height=64, width=64, count=4, dtype="float32", transform=transform
    ) as dst:
        dst.write(cube_data)

    with rasterio.open(
        ref_raster, "w", driver="GTiff", height=64, width=64, count=1, dtype="float32", transform=transform
    ) as dst:
        dst.write(np.ones((64, 64), dtype=np.float32) * 100.0, 1)

    source_product = MissionProduct(
        mission="Chandrayaan-2",
        instrument="IIRS",
        product_id="CH2_IIRS_TEST",
        image_path=source_raster,
        gsd_m=80.0,
        width=64,
        height=64,
        band_count=4,
        wavelengths=None,  # No wavelength metadata provided
    )
    ref_product = MissionProduct(
        mission="LRO",
        instrument="LROC NAC",
        product_id="LRO_NAC_TEST",
        image_path=ref_raster,
        gsd_m=1.0,
        width=64,
        height=64,
        band_count=1,
    )

    with pytest.raises(ValueError, match="IIRS_REPRESENTATION_UNCERTAIN"):
        extract_registration_windows(source_product, ref_product)

    # In register_products, it safely catches this and returns status IIRS_REPRESENTATION_UNCERTAIN
    out_dir = tmp_path / "out"
    result = register_products(source_product, ref_product, out_dir)
    assert result.status == "IIRS_REPRESENTATION_UNCERTAIN"
    assert "IIRS_REPRESENTATION_UNCERTAIN" in (result.failure_reason or "")
