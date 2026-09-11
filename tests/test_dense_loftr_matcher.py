"""
Comprehensive Test Suite for DenseLoFTRMatcher using kornia.feature.LoFTR and OpenCV.
SIH PS 26166: Multi-Modal, Sun-Angle, and Scale Invariant Registration.
"""

import cv2
import numpy as np
import pytest
import torch

from lunar_core.models import KeypointMatch
from lunar_core.alignment.dense_matcher import DenseLoFTRMatcher, DenseLoFTRResult
from lunar_core.data_io.synthetic_generator import LunarTerrainSimulator
from lunar_core.models import SunAngles


def test_prepare_geotiff_arrays():
    """Verifies GeoTIFF ingestion handles 2D, multi-band 3D, and padding to multiples of 8."""
    # 2D GeoTIFF with odd dimensions
    img_2d = np.random.uniform(10.0, 250.0, (125, 127)).astype(np.float32)
    tensor_2d, norm_2d = DenseLoFTRMatcher.prepare_geotiff_array(img_2d)
    assert tensor_2d.shape == (1, 1, 128, 128)  # Padded to multiples of 8
    assert norm_2d.shape == (125, 127)
    assert 0.0 <= norm_2d.min() and norm_2d.max() <= 1.0

    # 3D multi-band GeoTIFF [3, H, W]
    img_3d = np.random.uniform(0.0, 1.0, (3, 64, 64)).astype(np.float32)
    tensor_3d, norm_3d = DenseLoFTRMatcher.prepare_geotiff_array(img_3d)
    assert tensor_3d.shape == (1, 1, 64, 64)
    assert norm_3d.shape == (64, 64)


def test_grid_based_anms_8x8():
    """Verifies ANMS distributes matches uniformly across an 8x8 grid without crater rim clumping."""
    matcher = DenseLoFTRMatcher(confidence_threshold=0.20, grid_bins=8, cap_per_cell=3)

    # Create 30 matches all clumped into cell (0, 0) and 10 in cell (7, 7)
    clumped_matches: list[KeypointMatch] = []
    for i in range(30):
        clumped_matches.append(
            KeypointMatch(
                ref_xy=(5.0, 5.0),
                target_xy=(5.0, 5.0),
                confidence=0.95 - i * 0.01,
            )
        )
    for i in range(10):
        clumped_matches.append(
            KeypointMatch(
                ref_xy=(250.0, 250.0),
                target_xy=(250.0, 250.0),
                confidence=0.90 - i * 0.01,
            )
        )

    capped_src = matcher.apply_grid_anms_8x8(clumped_matches, image_shape=(256, 256), use_source_coords=True)
    assert len(capped_src) == 6
    assert capped_src[0].confidence == 0.95

    capped_ref = matcher.apply_grid_anms_8x8(clumped_matches, image_shape=(256, 256), use_source_coords=False)
    assert len(capped_ref) == 6
    assert capped_ref[0].confidence == 0.95


def test_subpixel_taylor_2d_parabolic_refinement():
    """Verifies 2D parabolic Taylor-series interpolation refines continuous sub-pixel coordinates."""
    matcher = DenseLoFTRMatcher(patch_radius=4)

    # Synthetic reference and shifted source
    sim = LunarTerrainSimulator(size=(64, 64), seed=99)
    dem = sim.generate_dem(num_craters=4)
    img_ref = sim.render_optical_image(dem, SunAngles(azimuth_deg=45.0, elevation_deg=30.0))

    # Known sub-pixel shift: dx = +0.35, dy = -0.25
    trans = np.float32([[1, 0, 0.35], [0, 1, -0.25]])
    img_src = cv2.warpAffine(img_ref, trans, (64, 64))

    # Integer match at (30, 30)
    match = KeypointMatch(ref_xy=(30.0, 30.0), target_xy=(30.0, 30.0), confidence=0.9)
    refined = matcher.refine_subpixel_taylor_2d([match], img_src, img_ref)

    assert len(refined) == 1
    # Subpixel refined flag should be True
    assert refined[0].subpixel_refined
    # Shift should be close to the applied physical shift
    dx_est = refined[0].target_xy[0] - match.target_xy[0]
    dy_est = refined[0].target_xy[1] - match.target_xy[1]
    assert abs(dx_est - 0.35) < 0.25
    assert abs(dy_est - (-0.25)) < 0.25


def test_dense_loftr_end_to_end_matching_and_magsac():
    """Verifies LoFTR dense keypoint extraction, ANMS, subpixel refinement, USAC_MAGSAC and warping."""
    sim = LunarTerrainSimulator(size=(192, 192), seed=42)
    sun = SunAngles(azimuth_deg=60.0, elevation_deg=25.0)
    dem = sim.generate_dem(num_craters=8)
    img_ref = sim.render_optical_image(dem, sun)

    # Apply planar homography transformation (shift and slight scale)
    true_H = np.array([
        [1.02, 0.01, -3.5],
        [-0.01, 1.01, 2.8],
        [0.00001, 0.00002, 1.0]
    ], dtype=np.float32)
    img_src = cv2.warpPerspective(img_ref, np.linalg.inv(true_H), (192, 192))

    matcher = DenseLoFTRMatcher(
        pretrained="outdoor",
        confidence_threshold=0.15,
        grid_bins=8,
        cap_per_cell=4,
    )

    # Test tuple unpacking syntax
    inliers, H, warped_src = matcher.match(source_image=img_src, reference_image=img_ref)

    assert len(inliers) >= 4, f"Expected at least 4 inliers, got {len(inliers)}"
    assert H is not None, "Homography matrix H must not be None"
    assert H.shape == (3, 3)
    assert warped_src is not None
    assert warped_src.shape == img_ref.shape

    # Check that majority of inliers are subpixel refined and all have valid residuals
    subpixel_count = sum(1 for m in inliers if m.subpixel_refined)
    assert subpixel_count / len(inliers) > 0.70, f"Expected > 70% subpixel refined, got {subpixel_count}/{len(inliers)}"
    for m in inliers:
        assert m.residual_error is not None
        assert m.residual_error < 3.0

    # Mean reprojection error on inliers should be sub-pixel
    residuals = [m.residual_error for m in inliers]
    mean_rmse = np.sqrt(np.mean(np.array(residuals)**2))
    assert mean_rmse < 1.0, f"Expected sub-pixel inlier RMSE < 1.0 px, got {mean_rmse:.4f}"
