"""
Robust Model Estimation: USAC-MAGSAC++ and Thin-Plate Splines (TPS).
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import cv2
import numpy as np

from lunar_core.models import KeypointMatch, TransformationType


class RobustEstimator:
    """
    USAC-MAGSAC++ robust transformation estimator and outlier rejection engine.
    """

    def __init__(self, threshold_pixels: float = 1.5, max_iters: int = 5000, confidence: float = 0.999) -> None:
        self.thresh = threshold_pixels
        self.max_iters = max_iters
        self.conf = confidence

    def estimate(
        self,
        matches: List[KeypointMatch],
        model: TransformationType = TransformationType.HOMOGRAPHY,
    ) -> Tuple[Optional[np.ndarray], List[KeypointMatch]]:
        if len(matches) < 4:
            return None, []

        if any(m.source_frame != "FULL_SOURCE_IMAGE" or m.reference_frame != "FULL_REFERENCE_IMAGE" for m in matches):
            raise ValueError("MIXED_COORDINATE_FRAMES")

        # Authoritative convention: source FULL_IMAGE -> reference FULL_IMAGE.
        src_pts = np.array([m.target_xy for m in matches], dtype=np.float32)
        dst_pts = np.array([m.ref_xy for m in matches], dtype=np.float32)

        method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)

        if model == TransformationType.HOMOGRAPHY:
            matrix, mask = cv2.findHomography(
                src_pts, dst_pts, method=method,
                ransacReprojThreshold=self.thresh,
                maxIters=self.max_iters,
                confidence=self.conf,
            )
            m_3x3 = matrix
        else:
            matrix, mask = cv2.estimateAffine2D(
                src_pts, dst_pts, method=method,
                ransacReprojThreshold=self.thresh,
                maxIters=self.max_iters,
                confidence=self.conf,
            )
            m_3x3 = np.vstack([matrix, [0, 0, 1]]) if matrix is not None else None

        if matrix is None or mask is None:
            return None, []

        inlier_indices = np.where(mask.ravel() == 1)[0]
        inliers: List[KeypointMatch] = []

        for idx in inlier_indices:
            m = matches[idx]
            pt = np.array([m.target_xy[0], m.target_xy[1], 1.0], dtype=np.float64)
            proj = m_3x3 @ pt
            if abs(proj[2]) > 1e-7:
                proj_xy = (proj[0] / proj[2], proj[1] / proj[2])
            else:
                proj_xy = (proj[0], proj[1])

            res = float(np.sqrt((proj_xy[0] - m.ref_xy[0])**2 + (proj_xy[1] - m.ref_xy[1])**2))
            inliers.append(
                KeypointMatch(
                    ref_xy=m.ref_xy,
                    target_xy=m.target_xy,
                    confidence=m.confidence,
                    subpixel_refined=m.subpixel_refined,
                    residual_error=res,
                    source_frame=m.source_frame,
                    reference_frame=m.reference_frame,
                )
            )

        return matrix, inliers

    @staticmethod
    def warp_target_to_reference(
        image: np.ndarray,
        matrix: np.ndarray,
        model: TransformationType,
        output_shape: Tuple[int, int],
    ) -> np.ndarray:
        h, w = output_shape
        # Matrix convention is source/target -> reference. OpenCV performs inverse
        # mapping internally when WARP_INVERSE_MAP is not supplied.
        if model == TransformationType.HOMOGRAPHY:
            return cv2.warpPerspective(image, matrix, (w, h), flags=cv2.INTER_LINEAR)
        else:
            m_3x3 = np.vstack([matrix, [0, 0, 1]]) if matrix.shape == (2, 3) else matrix
            return cv2.warpAffine(image, m_3x3[:2, :], (w, h), flags=cv2.INTER_LINEAR)
