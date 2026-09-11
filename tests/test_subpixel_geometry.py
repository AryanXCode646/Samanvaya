import cv2
import numpy as np
import pytest

from lunar_core.models import KeypointMatch, TransformationType
from lunar_core.postprocessing.magsac import RobustEstimator
from samanvaya.registration.transform import RegistrationTransform


def test_subpixel_transform_consistency():
    """
    Test subpixel correspondence consistency under explicit subpixel fractional shifts.
    Offered offsets: 0.10, 0.25, 0.40, 0.70.
    Verify that subpixel points -> transform -> predicted reference match within 0.01 px.
    Document: Coordinates are 0-indexed column, row floating points relative to top-left (0, 0).
    """
    offsets = [0.10, 0.25, 0.40, 0.70]
    base_points = [
        (10.0, 15.0),
        (50.0, 15.0),
        (10.0, 65.0),
        (50.0, 65.0),
        (30.0, 40.0),
    ]

    # Known affine transform: rotation=2.0 deg, translation=(4.35, -2.15)
    center = (30.0, 40.0)
    rot_mat = cv2.getRotationMatrix2D(center, 2.0, 1.0)
    rot_mat[0, 2] += 4.35
    rot_mat[1, 2] += -2.15

    src_subpixel = []
    ref_subpixel = []

    for i, (bx, by) in enumerate(base_points):
        # Inject fractional subpixel offset
        frac = offsets[i % len(offsets)]
        sx = bx + frac
        sy = by + (1.0 - frac)
        src_subpixel.append((sx, sy))

        # Project via ground truth affine
        pt_h = np.array([sx, sy, 1.0], dtype=np.float64)
        rx = float(rot_mat[0] @ pt_h)
        ry = float(rot_mat[1] @ pt_h)
        ref_subpixel.append((rx, ry))

    matches = [
        KeypointMatch(
            ref_xy=r,
            target_xy=s,
            confidence=0.99,
            subpixel_refined=True,
            source_frame="FULL_SOURCE_IMAGE",
            reference_frame="FULL_REFERENCE_IMAGE",
        )
        for s, r in zip(src_subpixel, ref_subpixel)
    ]

    estimator = RobustEstimator(threshold_pixels=0.05)
    estimated_mat, inliers = estimator.estimate(matches, TransformationType.AFFINE)
    assert estimated_mat is not None
    assert len(inliers) == len(matches)

    reg_transform = RegistrationTransform("AFFINE", "FULL_SOURCE_IMAGE", "FULL_REFERENCE_IMAGE", estimated_mat, "USAC")

    src_arr = np.array(src_subpixel, dtype=np.float64)
    ref_arr = np.array(ref_subpixel, dtype=np.float64)
    pred_arr = reg_transform.apply_source_to_reference(src_arr)

    # Subpixel error should be tiny (< 0.01 px)
    errors = np.linalg.norm(pred_arr - ref_arr, axis=1)
    assert np.max(errors) < 0.01
    assert np.mean(errors) < 0.005
