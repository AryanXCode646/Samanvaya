import numpy as np
import pytest

from lunar_core.models import KeypointMatch, TransformationType
from lunar_core.postprocessing.magsac import RobustEstimator
from samanvaya.registration.coordinates import CoordinateAudit
from samanvaya.registration.transform import RegistrationTransform


def test_tiled_registration_and_single_origin_counting():
    """
    Simulate a 2x2 grid of tiles with 20px overlap margins.
    Known global transform: translation=(14.2, -9.8), rotation=0.0.
    Generate correspondences in each tile, convert tile -> full image,
    and verify the global transform without double-counting tile origins.
    """
    # Global transform: T(source) = reference = source + (14.2, -9.8)
    dx, dy = 14.2, -9.8

    tiles = [
        {"tile_id": 0, "col": 0.0, "row": 0.0, "w": 100, "h": 100},
        {"tile_id": 1, "col": 80.0, "row": 0.0, "w": 100, "h": 100},    # 20px overlap
        {"tile_id": 2, "col": 0.0, "row": 80.0, "w": 100, "h": 100},    # 20px overlap
        {"tile_id": 3, "col": 80.0, "row": 80.0, "w": 100, "h": 100},  # 20px overlap
    ]

    all_matches = []
    ground_truth_full_sources = []
    ground_truth_full_refs = []

    for tile in tiles:
        audit = CoordinateAudit(window_col_off=0.0, window_row_off=0.0, tile_col_off=tile["col"], tile_row_off=tile["row"])
        # Generate 4 points local to each tile
        tile_source_pts = [
            (25.0, 25.0),
            (75.0, 25.0),
            (25.0, 75.0),
            (75.0, 75.0),
        ]
        for tx, ty in tile_source_pts:
            full_src_x, full_src_y = audit.tile_to_full(tx, ty)
            full_ref_x = full_src_x + dx
            full_ref_y = full_src_y + dy

            # Verify roundtrip
            loc_x, loc_y = audit.full_to_tile(full_src_x, full_src_y)
            assert abs(loc_x - tx) < 1e-5
            assert abs(loc_y - ty) < 1e-5

            ground_truth_full_sources.append((full_src_x, full_src_y))
            ground_truth_full_refs.append((full_ref_x, full_ref_y))

            all_matches.append(
                KeypointMatch(
                    ref_xy=(full_ref_x, full_ref_y),
                    target_xy=(full_src_x, full_src_y),
                    confidence=0.95,
                    source_frame="FULL_SOURCE_IMAGE",
                    reference_frame="FULL_REFERENCE_IMAGE",
                )
            )

    # Estimate global transform from the consolidated tiled matches
    estimator = RobustEstimator(threshold_pixels=0.1)
    estimated_mat, inliers = estimator.estimate(all_matches, TransformationType.AFFINE)
    assert estimated_mat is not None
    assert len(inliers) == 16

    reg_transform = RegistrationTransform("AFFINE", "FULL_SOURCE_IMAGE", "FULL_REFERENCE_IMAGE", estimated_mat, "USAC")

    # Verify global transform reproduces full coordinates exactly
    sources_arr = np.array(ground_truth_full_sources, dtype=np.float32)
    expected_refs = np.array(ground_truth_full_refs, dtype=np.float32)
    pred_refs = reg_transform.apply_source_to_reference(sources_arr)

    np.testing.assert_allclose(pred_refs, expected_refs, atol=1e-3)
    np.testing.assert_allclose(estimated_mat[0, 2], dx, atol=1e-3)
    np.testing.assert_allclose(estimated_mat[1, 2], dy, atol=1e-3)
