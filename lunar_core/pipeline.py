"""
Unified Clean Architecture Pipeline Facade for Lunar Core.
"""

from __future__ import annotations

import time
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
from lunar_core.postprocessing.anms import SpatialUniformDistributor
from lunar_core.postprocessing.subpixel import AnalyticalSubpixelRefiner
from lunar_core.postprocessing.magsac import RobustEstimator
from lunar_core.evaluation.metrics import EvaluationEngine


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
        initial_matches = self.dense_matcher.match_patches(roi.ref_roi, roi.target_roi)

        # Map ROI matches back to global reference image space
        xmin, ymin = roi.ref_bbox[0], roi.ref_bbox[1]
        global_matches: List[KeypointMatch] = [
            KeypointMatch(
                ref_xy=(m.ref_xy[0] + xmin, m.ref_xy[1] + ymin),
                target_xy=(m.target_xy[0] + xmin, m.target_xy[1] + ymin),
                confidence=m.confidence,
            )
            for m in initial_matches
        ]

        # Step 5: Postprocessing - Grid-Based ANMS Equal-Cell Capping
        if self.enable_anms and global_matches:
            allocated_matches = self.anms.cap_grid_cells(global_matches, ref_image.shape, cap_per_cell=4)
        else:
            allocated_matches = global_matches

        # Step 6: Robust USAC-MAGSAC++ Estimation on Coarse Inliers
        matrix, inliers = self.estimator.estimate(allocated_matches, self.trans_type)

        # Step 7: Sub-Pixel Peak Refinement on Invariant Phase Congruency Surfaces
        if self.enable_subpixel and inliers:
            refined_inliers = self.subpixel_refiner.refine_matches_batch(
                inliers, pc_ref.max_moment, pc_tgt.max_moment, patch_radius=6
            )
            src_pts = np.array([m.ref_xy for m in refined_inliers], dtype=np.float32)
            dst_pts = np.array([m.target_xy for m in refined_inliers], dtype=np.float32)

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
                        final_mat, _ = cv2.estimateAffine2D(final_src, final_dst)
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
                                )
                                for k, idx in enumerate(sub_indices)
                            ]
                            matrix = refined_mat
            elif self.trans_type == TransformationType.HOMOGRAPHY:
                refined_mat, _ = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=1.5)
                if refined_mat is not None:
                    matrix = refined_mat
                    inliers = refined_inliers


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
            processing_time_ms=elapsed_ms,
        )

        return RegistrationResult(
            transformation_type=self.trans_type,
            transform_matrix=matrix,
            matches=global_matches,
            inliers=inliers,
            metrics=metrics,
            warped_target=warped,
        )
