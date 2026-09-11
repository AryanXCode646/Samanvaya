import numpy as np
import pytest

from lunar_core.models import KeypointMatch, TransformationType
from lunar_core.postprocessing.magsac import RobustEstimator
from samanvaya.registration.transform import RegistrationTransform


def test_source_to_reference_direction_and_inversion_detection():
    """
    Ensure the estimator determines T(source) = reference.
    Verify that estimating inverted direction (reference -> source)
    and applying it directly to source raster leads to double displacement error.
    """
    # Define asymmetric shift: dx=+12.0, dy=-7.0
    dx, dy = 12.0, -7.0
    source_pts = np.array([[20.0, 30.0], [80.0, 30.0], [20.0, 90.0], [80.0, 90.0], [50.0, 60.0]], dtype=np.float32)
    # reference = source + (dx, dy)
    ref_pts = source_pts + np.array([dx, dy], dtype=np.float32)

    # Correct matches: target_xy=source, ref_xy=reference
    correct_matches = [
        KeypointMatch(
            ref_xy=(float(r[0]), float(r[1])),
            target_xy=(float(s[0]), float(s[1])),
            confidence=1.0,
            source_frame="FULL_SOURCE_IMAGE",
            reference_frame="FULL_REFERENCE_IMAGE",
        )
        for s, r in zip(source_pts, ref_pts)
    ]

    mat, inliers = RobustEstimator(threshold_pixels=0.1).estimate(correct_matches, TransformationType.HOMOGRAPHY)
    assert mat is not None
    transform = RegistrationTransform("HOMOGRAPHY", "FULL_SOURCE_IMAGE", "FULL_REFERENCE_IMAGE", mat, "USAC")

    # Applying source -> reference must produce ref_pts
    pred_ref = transform.apply_source_to_reference(source_pts)
    np.testing.assert_allclose(pred_ref, ref_pts, atol=1e-3)

    # Warping an image with feature at (20, 30) must shift feature to (32, 23)
    img_source = np.zeros((120, 120), dtype=np.float32)
    img_source[30, 20] = 1.0  # (y=30, x=20)
    warped = RobustEstimator.warp_target_to_reference(img_source, mat, TransformationType.HOMOGRAPHY, (120, 120))
    wy, wx = np.unravel_index(np.argmax(warped), warped.shape)
    assert (wx, wy) == (32, 23)

    # REGRESSION TEST: Inverted transform estimation (reference -> source)
    inverted_matches = [
        KeypointMatch(
            ref_xy=(float(s[0]), float(s[1])),  # mistakenly flipped
            target_xy=(float(r[0]), float(r[1])),
            confidence=1.0,
            source_frame="FULL_SOURCE_IMAGE",
            reference_frame="FULL_REFERENCE_IMAGE",
        )
        for s, r in zip(source_pts, ref_pts)
    ]
    inv_mat, _ = RobustEstimator(threshold_pixels=0.1).estimate(inverted_matches, TransformationType.HOMOGRAPHY)
    assert inv_mat is not None

    # Warping with inverted matrix causes displacement in WRONG direction (-dx, -dy)
    bad_warped = RobustEstimator.warp_target_to_reference(img_source, inv_mat, TransformationType.HOMOGRAPHY, (120, 120))
    b_wy, b_wx = np.unravel_index(np.argmax(bad_warped), bad_warped.shape)
    # The feature lands at (20 - 12, 30 - (-7)) = (8, 37) rather than (32, 23)!
    assert (b_wx, b_wy) == (8, 37)
    assert (b_wx, b_wy) != (wx, wy)
