"""
Scale-Space Resampling, ROI Localization, and Multi-Step Scale Pyramids.
Includes HierarchicalMultiModalBridge bridging OHRC (0.25m) -> TMC-2 (5.0m) -> IIRS (80m).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, List, Optional, Tuple, Union
import cv2
import numpy as np

from lunar_core.alignment.fourier_mellin import FourierMellinAligner
from lunar_core.alignment.dense_matcher import DenseLoFTRMatcher
from lunar_core.preprocessing.spectral import HyperspectralBandSelector
from lunar_core.models import KeypointMatch

logger = logging.getLogger("lunar_core.scale_space")


@dataclass
class RoiBundle:
    ref_roi: np.ndarray
    target_roi: np.ndarray
    ref_bbox: Tuple[int, int, int, int]  # (xmin, ymin, xmax, ymax)
    coarse_rotation_deg: float
    coarse_scale: float
    coarse_translation: Tuple[float, float]
    confidence: float
    target_to_common: np.ndarray
    reference_common_to_full_scale: float = 1.0
    target_common_to_full_scale: float = 1.0
    reference_origin_full: Tuple[float, float] = (0.0, 0.0)
    target_origin_full: Tuple[float, float] = (0.0, 0.0)


@dataclass
class HierarchicalAlignmentResult:
    """
    Composite result of the 2-step cascade registration bridging a 320x scale ratio:
    OHRC (0.25m) -> TMC-2 (5.0m) -> IIRS (80m).
    """
    h_ohrc_to_tmc2: np.ndarray
    h_tmc2_to_iirs: np.ndarray
    h_ohrc_to_iirs: np.ndarray
    inliers_ohrc_tmc2: List[KeypointMatch]
    inliers_tmc2_iirs: List[KeypointMatch]
    iirs_structural_band: np.ndarray
    ohrc_gsd: float = 0.25
    tmc2_gsd: float = 5.0
    iirs_gsd: float = 80.0
    composite_scale_ratio: float = 320.0


class ScaleSpaceLocalizer:
    """
    Downsamples disparate sensor resolutions (e.g. 20x for OHRC vs TMC-2) and extracts overlapping ROIs.
    """

    @staticmethod
    def resample_to_common_gsd(
        ref_img: np.ndarray,
        ref_gsd: float,
        tgt_img: np.ndarray,
        tgt_gsd: float,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Anti-alias resamples the finer resolution image to match the coarser image GSD.
        """
        ratio = tgt_gsd / ref_gsd
        if np.isclose(ratio, 1.0, atol=1e-2):
            return ref_img.copy(), tgt_img.copy(), 1.0

        if ratio > 1.0:
            sigma = max(0.5, (ratio - 1.0) * 0.5)
            ksize = int(2 * np.ceil(2 * sigma) + 1)
            blurred = cv2.GaussianBlur(ref_img, (ksize, ksize), sigma)
            new_w = max(16, int(round(ref_img.shape[1] / ratio)))
            new_h = max(16, int(round(ref_img.shape[0] / ratio)))
            resampled = cv2.resize(blurred, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return resampled, tgt_img.copy(), ratio
        else:
            inv_ratio = 1.0 / ratio
            sigma = max(0.5, (inv_ratio - 1.0) * 0.5)
            ksize = int(2 * np.ceil(2 * sigma) + 1)
            blurred = cv2.GaussianBlur(tgt_img, (ksize, ksize), sigma)
            new_w = max(16, int(round(tgt_img.shape[1] / inv_ratio)))
            new_h = max(16, int(round(tgt_img.shape[0] / inv_ratio)))
            resampled = cv2.resize(blurred, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return ref_img.copy(), resampled, ratio

    @classmethod
    def extract_coarse_roi(
        cls,
        img_ref: np.ndarray,
        img_tgt: np.ndarray,
        ref_gsd: float = 1.0,
        tgt_gsd: float = 1.0,
        padding: int = 16,
    ) -> RoiBundle:
        ref_comm, tgt_comm, scale_ratio = cls.resample_to_common_gsd(img_ref, ref_gsd, img_tgt, tgt_gsd)
        rot_deg, scale_est, dx, dy, conf = FourierMellinAligner.estimate_coarse_similarity(ref_comm, tgt_comm)

        h_ref, w_ref = ref_comm.shape
        center = (tgt_comm.shape[1] / 2.0, tgt_comm.shape[0] / 2.0)
        rot_mat = cv2.getRotationMatrix2D(center, -rot_deg, 1.0 / (scale_est + 1e-6))
        rot_mat[0, 2] += dx
        rot_mat[1, 2] += dy

        aligned_tgt = cv2.warpAffine(tgt_comm, rot_mat, (w_ref, h_ref), flags=cv2.INTER_LINEAR)
        mask = (ref_comm > 0).astype(np.uint8) & (aligned_tgt > 0).astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(np.vstack(contours))
            xmin, ymin = max(0, x - padding), max(0, y - padding)
            xmax, ymax = min(w_ref, x + w + padding), min(h_ref, y + h + padding)
        else:
            xmin, ymin, xmax, ymax = 0, 0, w_ref, h_ref

        return RoiBundle(
            ref_roi=ref_comm[ymin:ymax, xmin:xmax],
            target_roi=aligned_tgt[ymin:ymax, xmin:xmax],
            ref_bbox=(xmin, ymin, xmax, ymax),
            coarse_rotation_deg=rot_deg,
            coarse_scale=scale_est * scale_ratio,
            coarse_translation=(dx, dy),
            confidence=conf,
            target_to_common=np.vstack([rot_mat, [0.0, 0.0, 1.0]]).astype(np.float64),
            reference_common_to_full_scale=scale_ratio if tgt_gsd > ref_gsd else 1.0,
            target_common_to_full_scale=(1.0 / scale_ratio) if tgt_gsd < ref_gsd else 1.0,
        )


class HierarchicalMultiModalBridge:
    """
    2-Step Multi-Modal Registration Cascade:
    OHRC (0.25m) -> TMC-2 (5.0m) -> IIRS (80m)

    Overcomes the severe 320x scale ratio and radiometric differences between panchromatic
    visible cameras and 256-channel infrared hyperspectral spectrometers.
    """

    def __init__(
        self,
        matcher: Optional[DenseLoFTRMatcher] = None,
        band_selector: Optional[HyperspectralBandSelector] = None,
    ) -> None:
        self.matcher = matcher or DenseLoFTRMatcher(
            pretrained="outdoor",
            confidence_threshold=0.15,
            grid_bins=8,
            cap_per_cell=4,
        )
        self.band_selector = band_selector or HyperspectralBandSelector()

    def align_cascade(
        self,
        ohrc_image: np.ndarray,
        tmc2_image: np.ndarray,
        iirs_cube: np.ndarray,
        ohrc_gsd: float = 0.25,
        tmc2_gsd: float = 5.0,
        iirs_gsd: float = 80.0,
        spectral_extraction_method: str = "continuum",
    ) -> HierarchicalAlignmentResult:
        """
        Executes hierarchical 2-step registration:
        Step 1: OHRC (0.25m) -> TMC-2 (5.0m) [20x scale ratio]
        Step 2: TMC-2 (5.0m) -> IIRS (80m)  [16x scale ratio]
        Compound: OHRC -> IIRS                [320x scale ratio]
        """
        # --- Preprocessing: Extract structural band from IIRS cube ---
        if iirs_cube.ndim == 3:
            iirs_structural = self.band_selector.extract_optimal_structural_band(
                iirs_cube, method=spectral_extraction_method
            )
        else:
            iirs_structural = iirs_cube.astype(np.float32)

        # Normalize images to [0.0, 1.0] float32
        p1, p99 = np.percentile(ohrc_image, 1.0), np.percentile(ohrc_image, 99.0)
        ohrc_norm = np.clip((ohrc_image - p1) / max(p99 - p1, 1e-5), 0.0, 1.0).astype(np.float32)

        p1, p99 = np.percentile(tmc2_image, 1.0), np.percentile(tmc2_image, 99.0)
        tmc2_norm = np.clip((tmc2_image - p1) / max(p99 - p1, 1e-5), 0.0, 1.0).astype(np.float32)

        p1, p99 = np.percentile(iirs_structural, 1.0), np.percentile(iirs_structural, 99.0)
        iirs_norm = np.clip((iirs_structural - p1) / max(p99 - p1, 1e-5), 0.0, 1.0).astype(np.float32)

        # --- STEP 1: OHRC (0.25m) -> TMC-2 (5.0m) ---
        scale_ratio_1 = tmc2_gsd / ohrc_gsd  # 20.0
        ohrc_resampled, _, _ = ScaleSpaceLocalizer.resample_to_common_gsd(
            ohrc_norm, ohrc_gsd, tmc2_norm, tmc2_gsd
        )

        inliers_step1, h_step1_resampled, _ = self.matcher.match(ohrc_resampled, tmc2_norm)

        # Map resampled OHRC coordinates to full OHRC coordinates
        # x_resampled = x_ohrc / scale_ratio_1  => x_tmc2 = H_res @ [x_ohrc / scale_ratio_1]
        scale_mat_1 = np.array([
            [1.0 / scale_ratio_1, 0.0, 0.0],
            [0.0, 1.0 / scale_ratio_1, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        if h_step1_resampled is not None:
            h_ohrc_to_tmc2 = h_step1_resampled.astype(np.float64) @ scale_mat_1
        else:
            # Fallback to pure isotropic scaling
            h_ohrc_to_tmc2 = scale_mat_1.copy()

        # Update inlier matches with original OHRC coordinates
        full_inliers_step1: List[KeypointMatch] = []
        for m in inliers_step1:
            full_inliers_step1.append(
                KeypointMatch(
                    ref_xy=m.ref_xy,
                    target_xy=(m.target_xy[0] * scale_ratio_1, m.target_xy[1] * scale_ratio_1),
                    confidence=m.confidence,
                    subpixel_refined=m.subpixel_refined,
                )
            )

        # --- STEP 2: TMC-2 (5.0m) -> IIRS (80m) ---
        scale_ratio_2 = iirs_gsd / tmc2_gsd  # 16.0
        tmc2_resampled, _, _ = ScaleSpaceLocalizer.resample_to_common_gsd(
            tmc2_norm, tmc2_gsd, iirs_norm, iirs_gsd
        )

        inliers_step2, h_step2_resampled, _ = self.matcher.match(tmc2_resampled, iirs_norm)

        scale_mat_2 = np.array([
            [1.0 / scale_ratio_2, 0.0, 0.0],
            [0.0, 1.0 / scale_ratio_2, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        if h_step2_resampled is not None:
            h_tmc2_to_iirs = h_step2_resampled.astype(np.float64) @ scale_mat_2
        else:
            h_tmc2_to_iirs = scale_mat_2.copy()

        full_inliers_step2: List[KeypointMatch] = []
        for m in inliers_step2:
            full_inliers_step2.append(
                KeypointMatch(
                    ref_xy=m.ref_xy,
                    target_xy=(m.target_xy[0] * scale_ratio_2, m.target_xy[1] * scale_ratio_2),
                    confidence=m.confidence,
                    subpixel_refined=m.subpixel_refined,
                )
            )

        # --- STEP 3: Composite Compound Projection (OHRC -> IIRS) ---
        # x_iirs = H_tmc2_to_iirs @ x_tmc2 = H_tmc2_to_iirs @ H_ohrc_to_tmc2 @ x_ohrc
        h_ohrc_to_iirs = h_tmc2_to_iirs @ h_ohrc_to_tmc2
        if abs(h_ohrc_to_iirs[2, 2]) > 1e-8:
            h_ohrc_to_iirs = h_ohrc_to_iirs / h_ohrc_to_iirs[2, 2]

        composite_scale = iirs_gsd / ohrc_gsd

        return HierarchicalAlignmentResult(
            h_ohrc_to_tmc2=h_ohrc_to_tmc2,
            h_tmc2_to_iirs=h_tmc2_to_iirs,
            h_ohrc_to_iirs=h_ohrc_to_iirs,
            inliers_ohrc_tmc2=full_inliers_step1,
            inliers_tmc2_iirs=full_inliers_step2,
            iirs_structural_band=iirs_structural,
            ohrc_gsd=ohrc_gsd,
            tmc2_gsd=tmc2_gsd,
            iirs_gsd=iirs_gsd,
            composite_scale_ratio=composite_scale,
        )

    @staticmethod
    def transform_points(
        points_xy: np.ndarray,
        homography: np.ndarray,
    ) -> np.ndarray:
        """
        Projects an (N, 2) array of 2D coordinates through a 3x3 homography.
        """
        pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, homography.astype(np.float32))
        return transformed.reshape(-1, 2)

    @staticmethod
    def warp_image_to_target(
        source_image: np.ndarray,
        homography: np.ndarray,
        target_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Warps a 2D source image into target coordinates using homography.
        """
        h_tgt, w_tgt = target_shape[:2]
        return cv2.warpPerspective(
            source_image.astype(np.float32),
            homography.astype(np.float32),
            (w_tgt, h_tgt),
            flags=cv2.INTER_LINEAR,
        )
