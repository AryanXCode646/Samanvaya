"""
Comprehensive Test Suite for lunar_core Clean Architecture Framework.
"""

from pathlib import Path
import numpy as np
import pytest

from lunar_core import (
    GeoRaster,
    SunAngles,
    KeypointMatch,
    RegistrationMetrics,
    SensorModality,
    TransformationType,
    LunarCorePipeline,
)
from lunar_core.data_io import PlanetaryRasterReader, PlanetaryRasterWriter
from lunar_core.preprocessing import (
    PhaseCongruencyEngine,
    DynamicContrastEqualizer,
    PhotometricNormalizer,
)
from lunar_core.alignment import (
    FourierMellinAligner,
    ScaleSpaceLocalizer,
    DenseTransformerMatcher,
)
from lunar_core.postprocessing import (
    SpatialUniformDistributor,
    AnalyticalSubpixelRefiner,
    RobustEstimator,
)
from lunar_core.evaluation import EvaluationEngine
from lunar_core.data_io.synthetic_generator import LunarTerrainSimulator


def test_preprocessing_phase_congruency():
    sim = LunarTerrainSimulator(size=(64, 64), seed=7)
    dem = sim.generate_dem(num_craters=4)
    img = sim.render_optical_image(dem, SunAngles(azimuth_deg=45.0, elevation_deg=30.0))

    pc = PhaseCongruencyEngine(num_scales=3, num_orientations=4)
    out = pc.compute(img)
    assert out.phase_congruency.shape == (64, 64)
    assert out.max_moment.shape == (64, 64)
    assert 0.0 <= np.max(out.max_moment) <= 1.0


def test_preprocessing_retinex_and_photometric():
    img = np.random.uniform(0.1, 0.9, (64, 64)).astype(np.float32)
    contrast = DynamicContrastEqualizer()
    equalized = contrast.process(img)
    assert equalized.shape == (64, 64)

    norm = PhotometricNormalizer()
    sun = SunAngles(azimuth_deg=60.0, elevation_deg=25.0)
    corrected, mask = norm.normalize(img, sun)
    assert corrected.shape == (64, 64)
    assert mask.dtype == bool


def test_alignment_fourier_mellin_and_scale_space():
    sim = LunarTerrainSimulator(size=(100, 100), seed=42)
    dem = sim.generate_dem(num_craters=6)
    img_ref = sim.render_optical_image(dem, SunAngles(azimuth_deg=45.0, elevation_deg=30.0))
    img_tgt = img_ref.copy()

    rot, scale, dx, dy, conf = FourierMellinAligner.estimate_coarse_similarity(img_ref, img_tgt)
    assert abs(rot) < 5.0
    assert 0.8 < scale < 1.2

    roi = ScaleSpaceLocalizer.extract_coarse_roi(img_ref, img_tgt, ref_gsd=0.25, tgt_gsd=5.0)
    assert roi.ref_roi.shape[0] > 0
    assert roi.target_roi.shape[0] > 0



def test_alignment_dense_transformer_matcher():
    patch = np.random.uniform(0.2, 0.8, (48, 48)).astype(np.float32)
    matcher = DenseTransformerMatcher(confidence_threshold=0.75, grid_stride=8)
    matches = matcher.match_patches(patch, patch)
    assert len(matches) > 0
    for m in matches:
        assert m.confidence >= 0.75


def test_postprocessing_anms_and_subpixel():
    allocator = SpatialUniformDistributor(grid_rows=4, grid_cols=4)
    matches = [
        KeypointMatch(ref_xy=(5.0, 5.0), target_xy=(5.0, 5.0), confidence=0.9 - i * 0.01)
        for i in range(20)
    ]
    capped = allocator.cap_grid_cells(matches, (64, 64), cap_per_cell=3)
    assert len(capped) <= 3

    # Test analytical 6-parameter quadratic surface
    # f(x, y) = -2*x^2 - 1.5*y^2 + 0.3*x*y + 0.5*x - 0.2*y + 10.0
    a, b, c, d, e, f = -2.0, -1.5, 0.3, 0.5, -0.2, 10.0
    patch_3x3 = np.zeros((3, 3), dtype=np.float64)
    for r, y in enumerate([-1.0, 0.0, 1.0]):
        for col, x in enumerate([-1.0, 0.0, 1.0]):
            patch_3x3[r, col] = a*x**2 + b*y**2 + c*x*y + d*x + e*y + f

    refiner = AnalyticalSubpixelRefiner()
    fit = refiner.fit_quadratic_surface(patch_3x3)
    assert fit is not None
    dx_est, dy_est, _ = fit

    # Theoretical maximum
    det_h = 4.0 * a * b - c**2
    dx_true = (-2.0 * b * d + c * e) / det_h
    dy_true = (-2.0 * a * e + c * d) / det_h
    assert abs(dx_est - dx_true) < 1e-4
    assert abs(dy_est - dy_true) < 1e-4


def test_evaluation_metrics_and_entropy():
    matches = [
        KeypointMatch(ref_xy=(10.0, 10.0), target_xy=(10.2, 9.8), confidence=0.9, residual_error=0.28),
        KeypointMatch(ref_xy=(50.0, 10.0), target_xy=(50.1, 9.9), confidence=0.88, residual_error=0.14),
        KeypointMatch(ref_xy=(10.0, 50.0), target_xy=(10.3, 50.2), confidence=0.85, residual_error=0.36),
        KeypointMatch(ref_xy=(50.0, 50.0), target_xy=(49.9, 50.1), confidence=0.92, residual_error=0.14),
    ]
    metrics = EvaluationEngine.evaluate(total_matches=10, inliers=matches, image_shape=(64, 64))
    assert metrics.inlier_count == 4
    assert metrics.inlier_ratio == 0.40
    assert metrics.rmse_pixels < 0.40
    assert metrics.meets_isro_mandate
    assert metrics.spatial_uniformity_entropy > 0.0


def test_end_to_end_lunar_core_pipeline():
    sim = LunarTerrainSimulator(size=(128, 128), seed=55)
    sun_m = SunAngles(azimuth_deg=60.0, elevation_deg=25.0)
    sun_a = SunAngles(azimuth_deg=240.0, elevation_deg=35.0)

    img_ref, img_tgt, true_mat, _ = sim.generate_registered_pair_with_ground_truth(
        sun_ref=sun_m,
        sun_tgt=sun_a,
        true_translation=(3.0, -2.0),
        true_rotation_deg=0.0,
    )

    pipeline = LunarCorePipeline(
        transformation_type=TransformationType.AFFINE,
        enable_photometric=True,
        enable_anms=True,
        enable_subpixel=True,
    )
    result = pipeline.register(img_ref, img_tgt, sun_m, sun_a)
    assert result.metrics.inlier_count >= 4
    assert result.metrics.rmse_pixels < 0.40
    assert result.metrics.meets_isro_mandate
