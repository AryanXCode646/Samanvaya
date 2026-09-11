import numpy as np
import pytest

from lunar_core.models import KeypointMatch, TransformationType
from lunar_core.postprocessing.magsac import RobustEstimator
from samanvaya.registration.coordinates import CoordinateAudit
from samanvaya.registration.transform import RegistrationTransform


def test_pyramid_multi_scale_coordinate_propagation():
    """
    Test pyramid scales: 1.0, 0.5, 0.25, 0.125.
    At each scale, generate known correspondences with both integer and subpixel coordinates.
    Convert back to full-image coordinates and evaluate predictions against ground truth.
    """
    pyramid_scales = [1.0, 0.5, 0.25, 0.125]
    global_shift = np.array([5.75, -3.25])

    all_matches = []
    test_sources = []
    test_expected_refs = []

    for scale in pyramid_scales:
        audit = CoordinateAudit(window_col_off=0.0, window_row_off=0.0, pyramid_scale=scale)

        # Points in pyramid space (both integer and fractional subpixel)
        pyr_points = [
            (12.0, 18.0),       # integer
            (34.25, 27.50),     # subpixel
            (15.10, 42.70),     # subpixel
            (50.0, 50.0),       # integer
        ]

        for px, py in pyr_points:
            # Map pyramid coordinate to full-image coordinate: x_full = x_pyr / scale
            full_src_x, full_src_y = audit.local_to_full(px, py)
            full_ref_x = full_src_x + global_shift[0]
            full_ref_y = full_src_y + global_shift[1]

            # Verify roundtrip
            recovered_px, recovered_py = audit.full_to_local(full_src_x, full_src_y)
            assert abs(recovered_px - px) < 1e-4
            assert abs(recovered_py - py) < 1e-4

            test_sources.append((full_src_x, full_src_y))
            test_expected_refs.append((full_ref_x, full_ref_y))

            all_matches.append(
                KeypointMatch(
                    ref_xy=(full_ref_x, full_ref_y),
                    target_xy=(full_src_x, full_src_y),
                    confidence=0.9,
                    subpixel_refined=(px != round(px) or py != round(py)),
                    source_frame="FULL_SOURCE_IMAGE",
                    reference_frame="FULL_REFERENCE_IMAGE",
                )
            )

    # Estimate global transform from the consolidated multi-scale matches
    estimator = RobustEstimator(threshold_pixels=0.1)
    estimated_mat, inliers = estimator.estimate(all_matches, TransformationType.AFFINE)
    assert estimated_mat is not None
    assert len(inliers) == len(all_matches)

    reg_transform = RegistrationTransform("AFFINE", "FULL_SOURCE_IMAGE", "FULL_REFERENCE_IMAGE", estimated_mat, "USAC")
    src_arr = np.array(test_sources, dtype=np.float32)
    expected_arr = np.array(test_expected_refs, dtype=np.float32)
    pred_arr = reg_transform.apply_source_to_reference(src_arr)

    # All predictions across all scales (1.0 down to 0.125) must match ground truth
    rmse = np.sqrt(np.mean(np.linalg.norm(pred_arr - expected_arr, axis=1) ** 2))
    assert rmse < 1e-3
    np.testing.assert_allclose(pred_arr, expected_arr, atol=1e-3)
