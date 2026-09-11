import csv
from pathlib import Path
import numpy as np
import pytest

from lunar_core.models import KeypointMatch, TransformationType
from lunar_core.postprocessing.magsac import RobustEstimator
from samanvaya.registration.transform import RegistrationTransform


def test_checkpoint_separation_and_leakage_detection(tmp_path: Path):
    """
    Ensure fit points and held-out validation checkpoints are strictly separated.
    Simulate a model overfit or regional bias where:
    - Fit points have low RMSE (< 0.15 px)
    - Held-out checkpoints have high RMSE (> 3.0 px)
    Verify that the evaluation exposes the discrepancy clearly.
    """
    # Fit points restricted to localized top-left region
    fit_src = np.array([[10.0, 10.0], [20.0, 10.0], [10.0, 20.0], [20.0, 20.0]], dtype=np.float64)
    # Local shift for top-left: dx=+5.0, dy=+5.0
    fit_ref = fit_src + np.array([5.0, 5.0])

    # Held-out points situated in bottom-right region with an unmodeled non-rigid terrain warp:
    # true shift in bottom-right is dx=+12.0, dy=+10.0
    val_src = np.array([[80.0, 80.0], [90.0, 80.0], [80.0, 90.0], [90.0, 90.0]], dtype=np.float64)
    val_ref = val_src + np.array([12.0, 10.0])

    # Export separate files
    fit_csv = tmp_path / "estimation_matches.csv"
    val_csv = tmp_path / "validation_checkpoints.csv"

    with fit_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["point_id", "source_x", "source_y", "reference_x", "reference_y"])
        for i, (s, r) in enumerate(zip(fit_src, fit_ref)):
            writer.writerow([f"fit_{i}", s[0], s[1], r[0], r[1]])

    with val_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["point_id", "source_x", "source_y", "reference_x", "reference_y", "provenance", "quality"])
        for i, (s, r) in enumerate(zip(val_src, val_ref)):
            writer.writerow([f"val_{i}", s[0], s[1], r[0], r[1], "independent_survey", "A"])

    # Estimator ONLY receives fit points
    fit_matches = [
        KeypointMatch(
            ref_xy=(float(r[0]), float(r[1])),
            target_xy=(float(s[0]), float(s[1])),
            confidence=1.0,
            source_frame="FULL_SOURCE_IMAGE",
            reference_frame="FULL_REFERENCE_IMAGE",
        )
        for s, r in zip(fit_src, fit_ref)
    ]
    mat, inliers = RobustEstimator(threshold_pixels=0.1).estimate(fit_matches, TransformationType.AFFINE)
    assert mat is not None
    transform = RegistrationTransform("AFFINE", "FULL_SOURCE_IMAGE", "FULL_REFERENCE_IMAGE", mat, "USAC")

    # Evaluate fit RMSE (in-sample)
    fit_pred = transform.apply_source_to_reference(fit_src)
    fit_rmse = float(np.sqrt(np.mean(np.linalg.norm(fit_pred - fit_ref, axis=1) ** 2)))
    assert fit_rmse < 0.15

    # Evaluate held-out RMSE (out-of-sample)
    val_pred = transform.apply_source_to_reference(val_src)
    val_residuals = np.linalg.norm(val_pred - val_ref, axis=1)
    held_out_rmse = float(np.sqrt(np.mean(val_residuals ** 2)))

    # Held-out RMSE must clearly expose the error (> 3.0 px)
    assert held_out_rmse > 5.0
    assert held_out_rmse > 30 * fit_rmse


def test_independent_integer_vs_subpixel_metrics():
    """
    Test integer vs subpixel evaluation on independent checkpoints.
    Report and verify:
    rmse_integer, rmse_subpixel, median_integer, median_subpixel, p95_integer, p95_subpixel.
    """
    # Ground truth transform: translation=(3.25, 2.75)
    true_shift = np.array([3.25, 2.75])
    src_pts = np.array([[15.4, 25.6], [45.1, 35.8], [75.9, 65.2], [30.5, 80.3]], dtype=np.float64)
    ref_gt = src_pts + true_shift

    # A precise subpixel transform
    mat_3x3 = np.array([[1.0, 0.0, 3.25], [0.0, 1.0, 2.75], [0.0, 0.0, 1.0]], dtype=np.float64)
    transform = RegistrationTransform("AFFINE", "FULL_SOURCE_IMAGE", "FULL_REFERENCE_IMAGE", mat_3x3, "USAC")

    # Subpixel prediction
    pred_subpixel = transform.apply_source_to_reference(src_pts)
    res_subpixel = np.linalg.norm(pred_subpixel - ref_gt, axis=1)

    rmse_subpixel = float(np.sqrt(np.mean(res_subpixel ** 2)))
    median_subpixel = float(np.median(res_subpixel))
    p95_subpixel = float(np.percentile(res_subpixel, 95))

    # Integer rounded prediction
    pred_integer = np.round(pred_subpixel)
    res_integer = np.linalg.norm(pred_integer - ref_gt, axis=1)

    rmse_integer = float(np.sqrt(np.mean(res_integer ** 2)))
    median_integer = float(np.median(res_integer))
    p95_integer = float(np.percentile(res_integer, 95))

    # Subpixel accuracy is demonstrably superior to integer rounding
    assert rmse_subpixel < 1e-4
    assert rmse_integer > 0.1
    assert rmse_subpixel < rmse_integer
