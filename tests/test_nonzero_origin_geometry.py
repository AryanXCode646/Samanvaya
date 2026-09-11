import numpy as np
import pytest

from lunar_core.models import KeypointMatch, TransformationType
from lunar_core.postprocessing.magsac import RobustEstimator
from samanvaya.registration.coordinates import CoordinateAudit
from samanvaya.registration.transform import RegistrationTransform


def test_nonzero_window_origins_recovery():
    """
    Construct non-zero source window origin (col=150, row=220) and
    different reference window origin (col=310, row=180).
    Verify that restoring window offsets produces the correct full-image transform,
    and demonstrate that failing to add window offsets fails the test.
    """
    source_audit = CoordinateAudit(window_col_off=150.0, window_row_off=220.0)
    ref_audit = CoordinateAudit(window_col_off=310.0, window_row_off=180.0)

    # Known full-image global transform: translation=(+160.0, -40.0)
    true_global_shift = np.array([160.0, -40.0])

    # Local window coordinates
    local_source_pts = np.array([[20.0, 30.0], [70.0, 30.0], [20.0, 80.0], [70.0, 80.0]], dtype=np.float32)

    # Convert local source points to full image:
    full_source_pts = np.array([source_audit.local_to_full(x, y) for x, y in local_source_pts], dtype=np.float32)

    # Expected full-image reference points
    full_ref_pts = full_source_pts + true_global_shift

    # The matcher saw points in reference window:
    local_ref_pts = np.array([ref_audit.full_to_local(x, y) for x, y in full_ref_pts], dtype=np.float32)

    # Notice: local_ref_pts - local_source_pts is (0.0, 0.0) because (310 - 150 = 160) and (180 - 220 = -40)!
    # If a naive implementation ignores window offsets, it estimates the identity transform:
    buggy_matches = [
        KeypointMatch(
            ref_xy=(float(lr[0]), float(lr[1])),
            target_xy=(float(ls[0]), float(ls[1])),
            confidence=1.0,
            source_frame="FULL_SOURCE_IMAGE",
            reference_frame="FULL_REFERENCE_IMAGE",
        )
        for ls, lr in zip(local_source_pts, local_ref_pts)
    ]
    buggy_mat, _ = RobustEstimator(threshold_pixels=0.1).estimate(buggy_matches, TransformationType.AFFINE)
    # The buggy matrix has shift ~ 0, NOT 160, -40
    assert abs(buggy_mat[0, 2]) < 1e-3
    assert abs(buggy_mat[1, 2]) < 1e-3

    # The correct implementation propagates offsets back to full image:
    corrected_matches = [
        KeypointMatch(
            ref_xy=ref_audit.local_to_full(float(lr[0]), float(lr[1])),
            target_xy=source_audit.local_to_full(float(ls[0]), float(ls[1])),
            confidence=1.0,
            source_frame="FULL_SOURCE_IMAGE",
            reference_frame="FULL_REFERENCE_IMAGE",
        )
        for ls, lr in zip(local_source_pts, local_ref_pts)
    ]
    correct_mat, inliers = RobustEstimator(threshold_pixels=0.1).estimate(corrected_matches, TransformationType.AFFINE)
    assert correct_mat is not None
    assert len(inliers) == 4

    reg_transform = RegistrationTransform("AFFINE", "FULL_SOURCE_IMAGE", "FULL_REFERENCE_IMAGE", correct_mat, "USAC")
    predicted_full_ref = reg_transform.apply_source_to_reference(full_source_pts)
    np.testing.assert_allclose(predicted_full_ref, full_ref_pts, atol=1e-3)
    np.testing.assert_allclose(correct_mat[0, 2], 160.0, atol=1e-3)
    np.testing.assert_allclose(correct_mat[1, 2], -40.0, atol=1e-3)
