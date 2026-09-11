"""
Unified Clean Architecture Pipeline Facade for Lunar Core.
"""

from __future__ import annotations

import time
import logging
from typing import List, Optional
import cv2
import numpy as np

from lunar_core.models import (
    KeypointMatch,
    RegistrationMetrics,
    RegistrationResult,
    SunAngles,
    TransformationType,
)
from lunar_core.preprocessing.phase_congruency import PhaseCongruencyEngine
from lunar_core.preprocessing.photometric import PhotometricNormalizer
from lunar_core.preprocessing.contrast import DynamicContrastEqualizer
from lunar_core.alignment.fourier_mellin import FourierMellinAligner
from lunar_core.alignment.scale_space import ScaleSpaceLocalizer
from lunar_core.alignment.dense_matcher import DenseTransformerMatcher
from lunar_core.alignment.rift_matcher import ClassicalRIFTMatcher
from lunar_core.postprocessing.anms import SpatialUniformDistributor
from lunar_core.postprocessing.subpixel import AnalyticalSubpixelRefiner
from lunar_core.postprocessing.magsac import RobustEstimator
from lunar_core.evaluation.metrics import EvaluationEngine

logger = logging.getLogger(__name__)


class LunarCorePipeline:
    """
    End-to-End Clean Architecture Engine for ISRO Chandrayaan-2 Planetary Image Registration.
    """

    def __init__(
        self,
        transformation_type: TransformationType = TransformationType.HOMOGRAPHY,
        enable_photometric: bool = True,
        enable_anms: bool = True,
        enable_subpixel: bool = True,
        temperature: float = 0.08,
        confidence_threshold: float = 0.75,
    ) -> None:
        self.trans_type = transformation_type
        self.enable_photometric = enable_photometric
        self.enable_anms = enable_anms
        self.enable_subpixel = enable_subpixel

        # Pipeline components
        self.photometric = PhotometricNormalizer()
        self.contrast = DynamicContrastEqualizer()
        self.pc_engine = PhaseCongruencyEngine(num_scales=3, num_orientations=4)
        self.dense_matcher = DenseTransformerMatcher(temperature=temperature, confidence_threshold=confidence_threshold)
        self.anms = SpatialUniformDistributor(grid_rows=8, grid_cols=8)
        self.subpixel_refiner = AnalyticalSubpixelRefiner()
        self.estimator = RobustEstimator(threshold_pixels=1.5)

    def register(
        self,
        ref_image: np.ndarray,
        target_image: np.ndarray,
        ref_sun: Optional[SunAngles] = None,
        target_sun: Optional[SunAngles] = None,
        ref_gsd: float = 1.0,
        target_gsd: float = 1.0,
        dem_data: Optional[np.ndarray] = None,
        ref_dem: Optional[np.ndarray] = None,
        target_dem: Optional[np.ndarray] = None,
    ) -> RegistrationResult:
        start_time = time.perf_counter()

        # Step 1: Preprocessing & Photometric Normalization with optional DEM slope correction
        if self.enable_photometric and ref_sun and target_sun:
            r_dem = ref_dem if ref_dem is not None else dem_data
            t_dem = target_dem if target_dem is not None else dem_data
            img_ref_norm, _ = self.photometric.normalize(
                ref_image, ref_sun, dem_data=r_dem, pixel_gsd=ref_gsd
            )
            img_tgt_norm, _ = self.photometric.normalize(
                target_image, target_sun, dem_data=t_dem, pixel_gsd=target_gsd
            )
        else:
            img_ref_norm = (ref_image - np.min(ref_image)) / (np.ptp(ref_image) + 1e-6)
            img_tgt_norm = (target_image - np.min(target_image)) / (np.ptp(target_image) + 1e-6)

        # Step 2: Illumination-Invariant Log-Gabor Phase Congruency
        pc_ref = self.pc_engine.compute(img_ref_norm)
        pc_tgt = self.pc_engine.compute(img_tgt_norm)

        # Step 3: Coarse Multi-Scale Fourier-Mellin ROI Extraction
        roi = ScaleSpaceLocalizer.extract_coarse_roi(
            pc_ref.max_moment, pc_tgt.max_moment, ref_gsd, target_gsd
        )

        # Step 4: Fine Dense Cross-Attention Matching on ROI
        try:
            initial_matches = self.dense_matcher.match_patches(roi.ref_roi, roi.target_roi)
        except RuntimeError as exc:
            err_text = str(exc).lower()
            if "out of memory" not in err_text and "cuda" not in err_text and "model_weights_unavailable" not in err_text:
                raise
            logger.warning("Dense LoFTR failed (%s); switching to Classical RIFT fallback.", exc)
            initial_matches = []

        # Map matcher coordinates from the common aligned ROI back to each original image.
        # The target ROI is not in the reference frame: undo coarse alignment first.
        xmin, ymin = roi.ref_bbox[0], roi.ref_bbox[1]
        target_from_common = np.linalg.inv(roi.target_to_common)

        def common_to_target_full(x: float, y: float) -> tuple[float, float]:
            point = target_from_common @ np.array([x + xmin, y + ymin, 1.0], dtype=np.float64)
            point /= point[2]
            return (
                float(point[0] * roi.target_common_to_full_scale),
                float(point[1] * roi.target_common_to_full_scale),
            )

        def common_to_reference_full(x: float, y: float) -> tuple[float, float]:
            return (
                float((x + xmin) * roi.reference_common_to_full_scale),
                float((y + ymin) * roi.reference_common_to_full_scale),
            )

        global_matches: List[KeypointMatch] = [
            KeypointMatch(
                ref_xy=common_to_reference_full(m.ref_xy[0], m.ref_xy[1]),
                target_xy=common_to_target_full(m.target_xy[0], m.target_xy[1]),
                confidence=m.confidence,
                source_frame="FULL_SOURCE_IMAGE",
                reference_frame="FULL_REFERENCE_IMAGE",
            )
            for m in initial_matches
        ]

        # Step 5: Postprocessing - Grid-Based ANMS Equal-Cell Capping in SOURCE Frame
        if self.enable_anms and global_matches:
            allocated_matches = self.anms.cap_grid_cells(global_matches, target_image.shape, cap_per_cell=4, use_source_coords=True)
        else:
            allocated_matches = global_matches

        # Step 6: Robust USAC-MAGSAC++ Estimation on Coarse Inliers
        matrix, inliers = self.estimator.estimate(allocated_matches, self.trans_type)
        matcher_path = "dense_loftr"
        if len(inliers) < 4:
            matcher_path = "classical_rift"
            logger.info("Dense LoFTR produced %d inliers; trying Classical RIFT fallback.", len(inliers))
            try:
                rift_matches = ClassicalRIFTMatcher().match(
                    pc_ref.max_moment,
                    pc_ref.orientation_max_idx,
                    pc_tgt.max_moment,
                    pc_tgt.orientation_max_idx,
                )
                global_matches = rift_matches
                allocated_matches = (
                    self.anms.cap_grid_cells(global_matches, target_image.shape, cap_per_cell=4, use_source_coords=True)
                    if self.enable_anms and global_matches
                    else global_matches
                )
                matrix, inliers = self.estimator.estimate(allocated_matches, self.trans_type)
                logger.info("Matcher path: %s (%d inliers from %d matches).", matcher_path, len(inliers), len(global_matches))
            except Exception:
                logger.exception("Classical RIFT fallback failed; returning an empty registration result.")
                matcher_path = "classical_rift_failed"
                matrix, inliers = None, []

        # Step 7: Sub-Pixel Peak Refinement on Invariant Phase Congruency Surfaces
        if self.enable_subpixel and inliers:
            refined_inliers = self.subpixel_refiner.refine_matches_batch(
                inliers, pc_ref.max_moment, pc_tgt.max_moment, patch_radius=6
            )
            src_pts = np.array([m.target_xy for m in refined_inliers], dtype=np.float32)
            dst_pts = np.array([m.ref_xy for m in refined_inliers], dtype=np.float32)

            if self.trans_type == TransformationType.AFFINE:
                refined_mat, _ = cv2.estimateAffine2D(src_pts, dst_pts)
                if refined_mat is not None:
                    src_h = np.hstack([src_pts, np.ones((len(src_pts), 1), dtype=np.float32)])
                    pred = src_h @ refined_mat.T
                    res = np.linalg.norm(pred - dst_pts, axis=1)
                    sub_mask = res <= 0.55
                    if np.sum(sub_mask) >= 4:
                        final_src = src_pts[sub_mask]
                        final_dst = dst_pts[sub_mask]
                        final_mat, _ = cv2.estimateAffine2D(final_src, final_dst, method=cv2.LMEDS)
                        if final_mat is not None:
                            refined_mat = final_mat
                            src_h_sub = np.hstack([final_src, np.ones((len(final_src), 1), dtype=np.float32)])
                            pred_sub = src_h_sub @ final_mat.T
                            res_sub = np.linalg.norm(pred_sub - final_dst, axis=1)
                            sub_indices = np.where(sub_mask)[0]
                            inliers = [
                                KeypointMatch(
                                    ref_xy=refined_inliers[idx].ref_xy,
                                    target_xy=refined_inliers[idx].target_xy,
                                    confidence=refined_inliers[idx].confidence,
                                    subpixel_refined=True,
                                    residual_error=float(res_sub[k]),
                                    source_frame="FULL_SOURCE_IMAGE",
                                    reference_frame="FULL_REFERENCE_IMAGE",
                                )
                                for k, idx in enumerate(sub_indices)
                            ]
                            matrix = refined_mat
            elif self.trans_type == TransformationType.HOMOGRAPHY:
                method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
                refined_mat, mask = cv2.findHomography(
                    src_pts, dst_pts, method=method, ransacReprojThreshold=1.5, maxIters=5000, confidence=0.999
                )
                if refined_mat is not None and mask is not None:
                    inlier_idx = np.where(mask.ravel() == 1)[0]
                    if len(inlier_idx) >= 4:
                        matrix = refined_mat
                        recomputed_inliers = []
                        for idx in inlier_idx:
                            m = refined_inliers[idx]
                            pt_h = np.array([m.target_xy[0], m.target_xy[1], 1.0], dtype=np.float64)
                            proj = refined_mat @ pt_h
                            if abs(proj[2]) > 1e-8:
                                proj_xy = (proj[0] / proj[2], proj[1] / proj[2])
                                res = float(np.sqrt((proj_xy[0] - m.ref_xy[0]) ** 2 + (proj_xy[1] - m.ref_xy[1]) ** 2))
                            else:
                                res = float(np.linalg.norm(np.array(m.ref_xy) - np.array(m.target_xy)))
                            recomputed_inliers.append(
                                KeypointMatch(
                                    ref_xy=m.ref_xy,
                                    target_xy=m.target_xy,
                                    confidence=m.confidence,
                                    subpixel_refined=True,
                                    residual_error=res,
                                    sigma_x=m.sigma_x,
                                    sigma_y=m.sigma_y,
                                    cov_xy=m.cov_xy,
                                    weight=m.weight,
                                    source_frame="FULL_SOURCE_IMAGE",
                                    reference_frame="FULL_REFERENCE_IMAGE",
                                )
                            )
                        inliers = recomputed_inliers

        # Step 8: Warping target into reference coordinate frame
        warped: Optional[np.ndarray] = None
        if matrix is not None:
            warped = self.estimator.warp_target_to_reference(
                img_tgt_norm, matrix, self.trans_type, ref_image.shape
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Step 9: Evaluation Metrics
        metrics = EvaluationEngine.evaluate(
            total_matches=len(global_matches),
            inliers=inliers,
            image_shape=ref_image.shape,
            homography=matrix,
            processing_time_ms=elapsed_ms,
        )

        return RegistrationResult(
            transformation_type=self.trans_type,
            transform_matrix=matrix,
            matches=global_matches,
            inliers=inliers,
            metrics=metrics,
            warped_target=warped,
            matcher_path=matcher_path,
        )
