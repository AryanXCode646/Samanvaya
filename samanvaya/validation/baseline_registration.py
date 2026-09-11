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

    # Evaluate RMSE
    if checkpoints and len(checkpoints) >= 4:
        cp_src = np.array([[cp["source_x"], cp["source_y"]] for cp in checkpoints], dtype=float)
        cp_ref = np.array([[cp["reference_x"], cp["reference_y"]] for cp in checkpoints], dtype=float)
        homogeneous = np.column_stack([cp_src, np.ones(len(cp_src))])
        projected = homogeneous @ H.T
        projected = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-12)
        errors = np.linalg.norm(projected - cp_ref, axis=1)
        rmse = float(np.sqrt(np.mean(errors ** 2)))
    else:
        # Compute inlier reprojection RMSE
        inlier_src = src_pts[np.array(inlier_mask) == 1].reshape(-1, 2)
        homogeneous = np.column_stack([inlier_src, np.ones(len(inlier_src))])
        projected = homogeneous @ H.T
        projected = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-12)
        errors = np.linalg.norm(projected - inlier_ref_pts, axis=1)
        rmse = float(np.sqrt(np.mean(errors ** 2)))

    return {
        "method": f"baseline_{descriptor_name.lower()}_ransac",
        "descriptor": descriptor_name,
        "status": "SUCCESS",
        "raw_match_count": len(good_matches),
        "inlier_count": inlier_count,
        "inlier_ratio": float(inlier_count / len(good_matches)),
        "rmse_pixels": rmse,
        "coverage_fraction": min(1.0, coverage),
        "transform_matrix": H.tolist(),
    }
