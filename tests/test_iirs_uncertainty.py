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


def test_iirs_cube_pca_and_continuum_representation(tmp_path: Path):
    source_raster = tmp_path / "ch2_iirs_valid.tif"
    ref_raster = tmp_path / "ref_valid.tif"
    transform = from_origin(0.0, 100.0, 1.0, 1.0)

    # Multi-band cube (4 bands) with varying structure
    cube_data = np.random.RandomState(42).uniform(10.0, 200.0, (4, 64, 64)).astype(np.float32)
    with rasterio.open(
        source_raster, "w", driver="GTiff", height=64, width=64, count=4, dtype="float32", transform=transform
    ) as dst:
        dst.write(cube_data)

    with rasterio.open(
        ref_raster, "w", driver="GTiff", height=64, width=64, count=1, dtype="float32", transform=transform
    ) as dst:
        dst.write(cube_data[0], 1)

    # 1. Test PCA method when wavelengths absent
    source_pca_product = MissionProduct(
        mission="Chandrayaan-2",
        instrument="IIRS",
        product_id="CH2_IIRS_PCA",
        image_path=source_raster,
        gsd_m=80.0,
        width=64,
        height=64,
        band_count=4,
        wavelengths=None,
    )
    ref_product = MissionProduct(
        mission="LRO",
        instrument="LROC NAC",
        product_id="LRO_NAC_TEST",
        image_path=ref_raster,
        gsd_m=80.0,
        width=64,
        height=64,
        band_count=1,
    )

    windows_pca = extract_registration_windows(source_pca_product, ref_product, spectral_method="pca")
    assert windows_pca.spectral_representation == "robust_pca_structural_band"
    assert windows_pca.source_image.shape == (64, 64)
    assert windows_pca.spectral_provenance["instrument"] == "IIRS"

    # 2. Test continuum method when wavelengths present
    source_cont_product = MissionProduct(
        mission="Chandrayaan-2",
        instrument="IIRS",
        product_id="CH2_IIRS_CONT",
        image_path=source_raster,
        gsd_m=80.0,
        width=64,
        height=64,
        band_count=4,
        wavelengths=[900.0, 1100.0, 1200.0, 1400.0],
    )
    windows_cont = extract_registration_windows(source_cont_product, ref_product)
    assert windows_cont.spectral_representation == "wavelength_continuum_band"
    assert windows_cont.source_image.shape == (64, 64)


def test_iirs_bad_band_filtering():
    from lunar_core.preprocessing.spectral import HyperspectralBandSelector

    selector = HyperspectralBandSelector(num_bands=5)
    # Create 5-band cube: band 0 has normal data, band 1 is >50% NaN, band 2 is zero-variance, bands 3 & 4 are valid
    cube = np.ones((5, 32, 32), dtype=np.float32) * 50.0
    cube[0] = np.random.RandomState(1).randn(32, 32)
    cube[1, :20, :] = np.nan  # > 50% NaN
    cube[2] = 100.0  # zero variance
    cube[3] = np.random.RandomState(2).randn(32, 32)
    cube[4] = np.random.RandomState(3).randn(32, 32)

    clean_cube, kept, meta = selector.filter_bad_bands_and_pixels(cube)
    assert 1 not in kept  # high NaN band filtered out
    assert 2 not in kept  # zero variance band filtered out
    assert 0 in kept and 3 in kept and 4 in kept
    assert clean_cube.shape[0] == 3

