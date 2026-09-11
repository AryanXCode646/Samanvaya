import cv2
import numpy as np
import pytest

from lunar_core.models import KeypointMatch, TransformationType
from lunar_core.postprocessing.magsac import RobustEstimator
from samanvaya.registration.transform import (
    RegistrationTransform,
    check_geometric_plausibility,
    check_spatial_model_mismatch,
)


def test_registration_transform_roundtrip_and_frame_enforcement():
    source = np.array([[10.0, 10.0], [50.0, 10.0], [10.0, 50.0], [50.0, 50.0]], dtype=np.float64)
    matrix = np.array([[1.0, 0.0, 7.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]])
    expected_ref = np.array([[17.0, 7.0], [57.0, 7.0], [17.0, 47.0], [57.0, 47.0]])

    transform = RegistrationTransform(
        model_type="AFFINE",
        source_frame="FULL_SOURCE_IMAGE",
        target_frame="FULL_REFERENCE_IMAGE",
        parameters=matrix,
        estimation_method="KNOWN_SYNTHETIC",
        fit_statistics={"inliers": 4},
    )

    # Valid frame calls
    pred_ref = transform.apply_source_to_reference(source, source_frame="FULL_SOURCE_IMAGE")
    np.testing.assert_allclose(pred_ref, expected_ref, atol=1e-5)

    recovered_source = transform.apply_reference_to_source(expected_ref, target_frame="FULL_REFERENCE_IMAGE")
    np.testing.assert_allclose(recovered_source, source, atol=1e-5)

    # Reject frame mismatches
    with pytest.raises(ValueError, match="MIXED_COORDINATE_FRAMES"):
        transform.apply_source_to_reference(source, source_frame="ROI_WINDOW")

    with pytest.raises(ValueError, match="MIXED_COORDINATE_FRAMES"):
        transform.apply_reference_to_source(expected_ref, target_frame="TILE_FRAME")

    # Inverse object
    inv_transform = transform.inverse()
    assert inv_transform.source_frame == "FULL_REFERENCE_IMAGE"
    assert inv_transform.target_frame == "FULL_SOURCE_IMAGE"
    assert inv_transform.direction == "reference->source"
    np.testing.assert_allclose(inv_transform.apply_source_to_reference(expected_ref), source, atol=1e-5)


def test_mixed_coordinate_frames_in_estimator():
    matches = [
        KeypointMatch(
            ref_xy=(float(x), float(y)),
            target_xy=(float(x), float(y)),
            confidence=1.0,
            source_frame="ROI_LOCAL",
            reference_frame="FULL_REFERENCE_IMAGE",
        )
        for x, y in ((0, 0), (10, 0), (0, 10), (10, 10))
    ]
    with pytest.raises(ValueError, match="MIXED_COORDINATE_FRAMES"):
        RobustEstimator().estimate(matches, TransformationType.HOMOGRAPHY)


def test_geometric_plausibility_gates():
    # Valid affine transform
    valid_mat = np.array([[1.02, 0.01, 15.0], [-0.01, 1.01, -8.0], [0.0, 0.0, 1.0]])
    t_valid = RegistrationTransform("AFFINE", "FULL_SOURCE_IMAGE", "FULL_REFERENCE_IMAGE", valid_mat, "USAC")
    is_ok, reason = check_geometric_plausibility(t_valid)
    assert is_ok is True
    assert reason == "PLAUSIBLE"

    # Degenerate: negative determinant (reflection/inversion)
    neg_det_mat = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    t_neg = RegistrationTransform("AFFINE", "FULL_SOURCE_IMAGE", "FULL_REFERENCE_IMAGE", neg_det_mat, "USAC")
    is_ok, reason = check_geometric_plausibility(t_neg)
    assert is_ok is False
    assert "IMPLAUSIBLE_TRANSFORM" in reason

    # Degenerate: extreme scale distortion (scale > 20)
    huge_scale = np.array([[50.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 1.0]])
    t_huge = RegistrationTransform("AFFINE", "FULL_SOURCE_IMAGE", "FULL_REFERENCE_IMAGE", huge_scale, "USAC")
    is_ok, reason = check_geometric_plausibility(t_huge)
    assert is_ok is False
    assert "IMPLAUSIBLE_TRANSFORM" in reason

    # Degenerate: extreme projective tilt
    bad_proj = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.2, 0.3, 1.0]])
    t_proj = RegistrationTransform("HOMOGRAPHY", "FULL_SOURCE_IMAGE", "FULL_REFERENCE_IMAGE", bad_proj, "USAC")
    is_ok, reason = check_geometric_plausibility(t_proj)
    assert is_ok is False
    assert "IMPLAUSIBLE_TRANSFORM" in reason
