import cv2
import numpy as np
import pytest

from lunar_core.models import KeypointMatch, TransformationType
from lunar_core.postprocessing.magsac import RobustEstimator
from samanvaya.registration.transform import RegistrationTransform


def _make_pattern(size: tuple[int, int], centers: list[tuple[float, float]]) -> np.ndarray:
    img = np.zeros(size, dtype=np.float32)
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w]
    for cx, cy in centers:
        sigma = 3.0
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        img += blob
    return np.clip(img, 0.0, 1.0)


def test_synthetic_known_transform_recovery():
    """
    Apply a known affine transform (translation + rotation + scale) to feature pattern.
    Run transform estimation through the authoritative path and verify estimated ≈ known.
    """
    size = (150, 150)
    feature_coords = [(35.0, 40.0), (105.0, 40.0), (45.0, 110.0), (115.0, 105.0), (75.0, 75.0)]
    source_img = _make_pattern(size, feature_coords)

    # Known ground truth transform: translation=(8.5, -5.2), rotation=4.0 deg, scale=1.03
    center = (size[1] / 2.0, size[0] / 2.0)
    true_rot = cv2.getRotationMatrix2D(center, 4.0, 1.03)
    true_rot[0, 2] += 8.5
    true_rot[1, 2] += -5.2
    true_mat_3x3 = np.vstack([true_rot, [0.0, 0.0, 1.0]])

    # Ground truth reference coordinates: p_ref = true_mat * [x, y, 1]
    src_pts = np.array(feature_coords, dtype=np.float32)
    src_h = np.hstack([src_pts, np.ones((len(src_pts), 1), dtype=np.float32)])
    expected_ref_pts = (src_h @ true_rot.T).astype(np.float32)

    # Build matches in the authoritative frame
    matches = [
        KeypointMatch(
            ref_xy=(float(ref[0]), float(ref[1])),
            target_xy=(float(src[0]), float(src[1])),
            confidence=0.98,
            source_frame="FULL_SOURCE_IMAGE",
            reference_frame="FULL_REFERENCE_IMAGE",
        )
        for src, ref in zip(src_pts, expected_ref_pts)
    ]

    # Robust estimation
    estimator = RobustEstimator(threshold_pixels=0.1)
    estimated_mat, inliers = estimator.estimate(matches, TransformationType.AFFINE)
    assert estimated_mat is not None
    assert len(inliers) >= 4

    reg_transform = RegistrationTransform(
        model_type="AFFINE",
        source_frame="FULL_SOURCE_IMAGE",
        target_frame="FULL_REFERENCE_IMAGE",
        parameters=estimated_mat,
        estimation_method="USAC_MAGSAC",
        fit_statistics={"inliers": len(inliers)},
    )

    # Verify estimated transform reproduces reference points within 0.05 pixels
    predicted_ref = reg_transform.apply_source_to_reference(src_pts)
    rmse = np.sqrt(np.mean(np.linalg.norm(predicted_ref - expected_ref_pts, axis=1) ** 2))
    assert rmse < 0.05

    # Check matrix element closeness
    np.testing.assert_allclose(estimated_mat, true_rot, atol=1e-2)

    # Test warping of source raster using the estimated transform
    warped = RobustEstimator.warp_target_to_reference(source_img, estimated_mat, TransformationType.AFFINE, size)
    assert warped.shape == size
    assert np.max(warped) > 0.5
