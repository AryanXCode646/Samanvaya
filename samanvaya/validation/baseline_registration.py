"""Standard classical registration baseline for comparative evaluation.

Compares standard classical feature descriptors (SIFT / ORB + RANSAC) on the
EXACT SAME source, reference, and validation checkpoints without phase congruency
or deep transformer priors.
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
import cv2


def run_baseline_registration(
    source: np.ndarray,
    reference: np.ndarray,
    checkpoints: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Execute standard SIFT (or ORB fallback) + RANSAC registration.

    Args:
        source: 2-D unnormalized source array.
        reference: 2-D unnormalized reference array.
        checkpoints: Optional independent checkpoints with source_x, source_y, reference_x, reference_y.

    Returns:
        Dictionary containing match count, inlier count, RMSE, coverage, and status.
    """
    def _to_uint8(arr: np.ndarray) -> np.ndarray:
        finite = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        min_v, max_v = float(np.min(finite)), float(np.max(finite))
        if max_v - min_v < 1e-6:
            return np.zeros(finite.shape, dtype=np.uint8)
        norm = (finite - min_v) / (max_v - min_v) * 255.0
        return np.clip(norm, 0, 255).astype(np.uint8)

    src_u8 = _to_uint8(source)
    ref_u8 = _to_uint8(reference)

    # Try SIFT first, fallback to ORB
    descriptor_name = "SIFT"
    try:
        detector = cv2.SIFT_create()
        kp1, des1 = detector.detectAndCompute(src_u8, None)
        kp2, des2 = detector.detectAndCompute(ref_u8, None)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
    except Exception:
        descriptor_name = "ORB"
        detector = cv2.ORB_create(nfeatures=2000)
        kp1, des1 = detector.detectAndCompute(src_u8, None)
        kp2, des2 = detector.detectAndCompute(ref_u8, None)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return {
            "method": f"baseline_{descriptor_name.lower()}_ransac",
            "descriptor": descriptor_name,
            "status": "NO_MATCHES",
            "raw_match_count": 0,
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "rmse_pixels": None,
            "coverage_fraction": 0.0,
            "transform_matrix": None,
        }

    # k-NN match with Lowe's ratio test
    knn_matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for m_pair in knn_matches:
        if len(m_pair) == 2 and m_pair[0].distance < 0.8 * m_pair[1].distance:
            good_matches.append(m_pair[0])

    if len(good_matches) < 4:
        return {
            "method": f"baseline_{descriptor_name.lower()}_ransac",
            "descriptor": descriptor_name,
            "status": "NO_CORRESPONDENCE",
            "raw_match_count": len(good_matches),
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "rmse_pixels": None,
            "coverage_fraction": 0.0,
            "transform_matrix": None,
        }

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inlier_mask = mask.ravel().tolist() if mask is not None else []
    inlier_count = int(sum(inlier_mask))

    if H is None or inlier_count < 4:
        return {
            "method": f"baseline_{descriptor_name.lower()}_ransac",
            "descriptor": descriptor_name,
            "status": "LOW_INLIER_RATIO",
            "raw_match_count": len(good_matches),
            "inlier_count": inlier_count,
            "inlier_ratio": inlier_count / max(1, len(good_matches)),
            "rmse_pixels": None,
            "coverage_fraction": 0.0,
            "transform_matrix": None,
        }

    # Calculate spatial coverage
    inlier_ref_pts = dst_pts[np.array(inlier_mask) == 1].reshape(-1, 2)
    h, w = reference.shape[:2]
    x_span = float(np.ptp(inlier_ref_pts[:, 0])) if len(inlier_ref_pts) > 1 else 0.0
    y_span = float(np.ptp(inlier_ref_pts[:, 1])) if len(inlier_ref_pts) > 1 else 0.0
    coverage = float((x_span * y_span) / max(1.0, w * h))

    import time

    # Evaluate residuals
    errors: np.ndarray
    if checkpoints and len(checkpoints) >= 4:
        cp_src = np.array([[cp["source_x"], cp["source_y"]] for cp in checkpoints], dtype=float)
        cp_ref = np.array([[cp["reference_x"], cp["reference_y"]] for cp in checkpoints], dtype=float)
        homogeneous = np.column_stack([cp_src, np.ones(len(cp_src))])
        projected = homogeneous @ H.T
        projected = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-12)
        errors = np.linalg.norm(projected - cp_ref, axis=1)
        evaluation_basis = "held_out_checkpoints"
    else:
        # Compute inlier reprojection residuals
        inlier_src = src_pts[np.array(inlier_mask) == 1].reshape(-1, 2)
        homogeneous = np.column_stack([inlier_src, np.ones(len(inlier_src))])
        projected = homogeneous @ H.T
        projected = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-12)
        errors = np.linalg.norm(projected - inlier_ref_pts, axis=1)
        evaluation_basis = "inlier_reprojection"

    rmse = float(np.sqrt(np.mean(errors ** 2))) if len(errors) > 0 else None
    median_err = float(np.median(errors)) if len(errors) > 0 else None
    p95_err = float(np.percentile(errors, 95)) if len(errors) > 0 else None

    return {
        "method": f"baseline_{descriptor_name.lower()}_ransac",
        "descriptor": descriptor_name,
        "status": "SUCCESS",
        "raw_match_count": len(good_matches),
        "inlier_count": inlier_count,
        "inlier_ratio": float(inlier_count / len(good_matches)),
        "rmse_pixels": rmse,
        "median_error_px": median_err,
        "p95_error_px": p95_err,
        "evaluation_basis": evaluation_basis,
        "coverage_fraction": min(1.0, coverage),
        "transform_matrix": H.tolist(),
    }


def compare_matchers(
    source: np.ndarray,
    reference: np.ndarray,
    checkpoints: Optional[list[dict[str, Any]]] = None,
    source_gsd: float = 1.0,
    reference_gsd: float = 1.0,
) -> dict[str, Any]:
    """Execute evidence-based comparison between classical baseline and Samanvaya."""
    import time
    from lunar_core.pipeline import LunarCorePipeline
    from lunar_core.models import TransformationType

    # 1. Classical Baseline
    t0 = time.perf_counter()
    baseline_result = run_baseline_registration(source, reference, checkpoints=checkpoints)
    baseline_time_ms = (time.perf_counter() - t0) * 1000.0
    baseline_result["runtime_ms"] = baseline_time_ms

    # 2. Samanvaya LoFTR + Phase Congruency
    t1 = time.perf_counter()
    pipeline = LunarCorePipeline(transformation_type=TransformationType.HOMOGRAPHY)
    samanvaya_res = pipeline.register(
        reference,
        source,
        ref_gsd=reference_gsd,
        target_gsd=source_gsd,
    )
    samanvaya_time_ms = (time.perf_counter() - t1) * 1000.0

    sam_raw = len(samanvaya_res.matches)
    sam_inliers = len(samanvaya_res.inliers)
    sam_ratio = float(samanvaya_res.metrics.inlier_ratio)
    sam_rmse = samanvaya_res.metrics.rmse_pixels

    # Checkpoint evaluation for Samanvaya if provided
    sam_median = None
    sam_p95 = None
    if checkpoints and len(checkpoints) >= 4 and samanvaya_res.transform_matrix is not None:
        cp_src = np.array([[cp["source_x"], cp["source_y"]] for cp in checkpoints], dtype=float)
        cp_ref = np.array([[cp["reference_x"], cp["reference_y"]] for cp in checkpoints], dtype=float)
        homo = np.column_stack([cp_src, np.ones(len(cp_src))])
        proj = homo @ samanvaya_res.transform_matrix.T
        proj = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-12)
        res_errs = np.linalg.norm(proj - cp_ref, axis=1)
        sam_rmse = float(np.sqrt(np.mean(res_errs ** 2)))
        sam_median = float(np.median(res_errs))
        sam_p95 = float(np.percentile(res_errs, 95))

    samanvaya_dict = {
        "method": "samanvaya_loftr_phase_congruency",
        "status": "SUCCESS" if sam_inliers >= 4 else "NO_CORRESPONDENCE",
        "raw_match_count": sam_raw,
        "inlier_count": sam_inliers,
        "inlier_ratio": sam_ratio,
        "rmse_pixels": sam_rmse,
        "median_error_px": sam_median,
        "p95_error_px": sam_p95,
        "coverage_fraction": samanvaya_res.metrics.spatial_uniformity_entropy,
        "runtime_ms": samanvaya_time_ms,
    }

    return {
        "comparison_table": {
            "baseline": baseline_result,
            "samanvaya": samanvaya_dict,
        },
        "summary": {
            "more_inliers": "samanvaya" if sam_inliers > baseline_result.get("inlier_count", 0) else "baseline",
            "higher_inlier_ratio": "samanvaya" if sam_ratio > baseline_result.get("inlier_ratio", 0) else "baseline",
        }
    }

