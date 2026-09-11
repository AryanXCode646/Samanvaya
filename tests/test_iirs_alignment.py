"""
Unit Tests for IIRS Hyperspectral Continuum Extraction and Hierarchical Multi-Modal Scale Bridging.
Verifies HyperspectralBandSelector and HierarchicalMultiModalBridge across 320x scale ratios.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from lunar_core.preprocessing.spectral import HyperspectralBandSelector
from lunar_core.preprocessing.spectral import SpectralCube
from lunar_core.alignment.scale_space import (
    HierarchicalMultiModalBridge,
    HierarchicalAlignmentResult,
    ScaleSpaceLocalizer,
)


def create_synthetic_iirs_cube(
    height: int = 64,
    width: int = 64,
    num_bands: int = 256,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a synthetic 256-band IIRS cube with realistic mineral absorption and thermal emission.
    Returns (cube_array, ground_truth_spatial_pattern).
    """
    np.random.seed(seed)
    x = np.linspace(0, 4 * np.pi, width, dtype=np.float32)
    y = np.linspace(0, 4 * np.pi, height, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    spatial_pattern = np.sin(xv) * np.cos(yv) * 0.3 + 0.5  # [0.2, 0.8]

    wavelengths = np.linspace(800.0, 5000.0, num_bands, dtype=np.float32)

    # Continuum reflectance (peaks around 1.0 - 1.25 µm, drops towards UV/NIR)
    continuum_spectrum = np.exp(-((wavelengths - 1150.0) ** 2) / (2 * 400.0**2))

    # Thermal emission (rises steeply above 2500 nm)
    thermal_spectrum = np.zeros_like(wavelengths)
    thermal_mask = wavelengths > 2500.0
    thermal_spectrum[thermal_mask] = ((wavelengths[thermal_mask] - 2500.0) / 2500.0) ** 2 * 2.0

    # Broadcast into 3D cube (num_bands, height, width)
    cube = np.zeros((num_bands, height, width), dtype=np.float32)
    for b in range(num_bands):
        refl = spatial_pattern * continuum_spectrum[b]
        therm = thermal_spectrum[b]
        noise = np.random.normal(0, 0.01, (height, width)).astype(np.float32)
        cube[b] = np.clip(refl + therm + noise, 0.0, None)

    return cube, spatial_pattern


class TestHyperspectralBandSelector:
    """Tests for continuum window isolation and PCA dimensionality reduction."""

    def test_spectral_cube_representation_records_wavelength_provenance(self):
        cube = SpectralCube(
            np.ones((4, 8, 8), dtype=np.float32),
            wavelengths=np.array([1.0, 1.1, 1.2, 1.3], dtype=np.float32),
            wavelength_units="um",
            wavelength_source="pds4_label",
            axis_order="bands,line,sample",
            metadata_provenance="fixture.xml",
        )
        provenance = cube.representation_provenance("pca", components=1)
        assert provenance["wavelength_source"] == "pds4_label"
        assert provenance["wavelength_range"][0] == pytest.approx(1.0)
        assert provenance["wavelength_range"][1] == pytest.approx(1.3)
        assert provenance["axis_order"] == "bands,line,sample"

    def test_spectral_cube_without_wavelength_metadata_is_explicit(self):
        cube = SpectralCube(np.ones((4, 8, 8), dtype=np.float32))
        provenance = cube.representation_provenance("band_mean")
        assert provenance["wavelength_source"] == "WAVELENGTH_METADATA_UNAVAILABLE"
        assert provenance["wavelength_range"] is None

    def test_default_initialization(self):
        selector = HyperspectralBandSelector()
        assert selector.num_bands == 256
        assert len(selector.wavelengths) == 256
        assert np.isclose(selector.wavelengths[0], 800.0)
        assert np.isclose(selector.wavelengths[-1], 5000.0)

    def test_micrometer_conversion(self):
        # Pass wavelengths in µm [0.8, ..., 5.0]
        wl_um = np.linspace(0.8, 5.0, 50)
        selector = HyperspectralBandSelector(wavelengths=wl_um)
        assert np.isclose(selector.wavelengths[0], 800.0)
        assert np.isclose(selector.wavelengths[-1], 5000.0)

    def test_continuum_band_extraction(self):
        cube, ground_truth = create_synthetic_iirs_cube(64, 64, 256, seed=123)
        selector = HyperspectralBandSelector()

        continuum = selector.extract_continuum_band(cube, min_nm=1000.0, max_nm=1250.0)

        assert continuum.shape == (64, 64)
        assert continuum.dtype == np.float32
        assert np.min(continuum) >= 0.0
        assert np.max(continuum) <= 1.0

        # Verify continuum is strongly correlated with the true reflectance pattern
        corr = np.corrcoef(continuum.ravel(), ground_truth.ravel())[0, 1]
        assert corr > 0.85

    def test_pca_structural_band_extraction(self):
        cube, ground_truth = create_synthetic_iirs_cube(64, 64, 256, seed=456)
        selector = HyperspectralBandSelector()

        pca_band = selector.extract_pca_structural_band(cube)

        assert pca_band.shape == (64, 64)
        assert pca_band.dtype == np.float32
        assert np.min(pca_band) >= 0.0
        assert np.max(pca_band) <= 1.0

        # PCA First Component should capture dominant spatial pattern
        corr = np.corrcoef(pca_band.ravel(), ground_truth.ravel())[0, 1]
        assert corr > 0.70

    def test_channel_last_cube_support(self):
        cube, _ = create_synthetic_iirs_cube(48, 48, 64, seed=789)
        # Transpose to (H, W, bands)
        cube_hwc = np.transpose(cube, (1, 2, 0))
        assert cube_hwc.shape == (48, 48, 64)

        selector = HyperspectralBandSelector(num_bands=64)
        band = selector.extract_continuum_band(cube_hwc)
        assert band.shape == (48, 48)


class TestHierarchicalMultiModalBridge:
    """Tests 2-step scale bridging: OHRC (0.25m) -> TMC-2 (5.0m) -> IIRS (80m)."""

    def test_cascade_hierarchical_alignment(self):
        # 1. Synthesize multi-resolution lunar scene
        # Master high-resolution feature (e.g. crater)
        np.random.seed(999)
        master_ohrc = np.zeros((240, 240), dtype=np.float32)
        cv2.circle(master_ohrc, (120, 120), 40, 0.8, -1)
        cv2.circle(master_ohrc, (120, 120), 20, 0.2, -1)
        noise = np.random.normal(0, 0.03, (240, 240)).astype(np.float32)
        master_ohrc = np.clip(master_ohrc + noise, 0.0, 1.0)

        # TMC-2: intermediate scale with slight shift (dx=2, dy=1)
        tmc2 = cv2.resize(master_ohrc, (80, 80), interpolation=cv2.INTER_AREA)
        M_shift = np.float32([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
        tmc2 = cv2.warpAffine(tmc2, M_shift, (80, 80), borderMode=cv2.BORDER_REFLECT)

        # IIRS: coarse hyperspectral cube with 64 bands
        iirs_cube = np.zeros((64, 48, 48), dtype=np.float32)
        base_iirs = cv2.resize(tmc2, (48, 48), interpolation=cv2.INTER_AREA)
        for b in range(64):
            iirs_cube[b] = base_iirs + np.random.normal(0, 0.02, (48, 48)).astype(np.float32)

        # 2. Run hierarchical bridge
        bridge = HierarchicalMultiModalBridge()
        result = bridge.align_cascade(
            ohrc_image=master_ohrc,
            tmc2_image=tmc2,
            iirs_cube=iirs_cube,
            ohrc_gsd=0.25,
            tmc2_gsd=5.0,
            iirs_gsd=80.0,
            spectral_extraction_method="continuum",
        )

        assert isinstance(result, HierarchicalAlignmentResult)
        assert result.h_ohrc_to_tmc2.shape == (3, 3)
        assert result.h_tmc2_to_iirs.shape == (3, 3)
        assert result.h_ohrc_to_iirs.shape == (3, 3)
        assert result.composite_scale_ratio == 320.0

        # Verify mathematical composition: H_compound == H_step2 @ H_step1 (up to scale)
        expected_compound = result.h_tmc2_to_iirs @ result.h_ohrc_to_tmc2
        expected_compound = expected_compound / expected_compound[2, 2]
        np.testing.assert_allclose(result.h_ohrc_to_iirs, expected_compound, rtol=1e-5, atol=1e-5)

        # 3. Test coordinate projection through composite homography
        test_pts = np.array([[120.0, 120.0], [50.0, 50.0]], dtype=np.float32)
        projected = bridge.transform_points(test_pts, result.h_ohrc_to_iirs)

        assert projected.shape == (2, 2)
        # Projected coordinates should land inside the coarse IIRS frame
        assert np.all(projected >= -10.0)
        assert np.all(projected <= 60.0)

        # 4. Test image warping directly to IIRS target shape
        warped_ohrc = bridge.warp_image_to_target(master_ohrc, result.h_ohrc_to_iirs, (48, 48))
        assert warped_ohrc.shape == (48, 48)
        assert np.isfinite(warped_ohrc).all()
