"""
Classical Phase-Congruency RIFT (Radiation-variation Insensitive Feature Transform) Matcher.
Provides guaranteed CPU/illumination fallback when deep neural matchers (LoFTR/RoMa)
are unavailable or fail to produce sufficient matches.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple
import cv2
import numpy as np

from lunar_core.models import KeypointMatch

logger = logging.getLogger('lunar_core.rift_matcher')


class RIFTDescriptorExtractor:
    """
    Extracts Radiation-variation Insensitive Feature Transform (RIFT) descriptors.
    Uses Maximum Index Map (MIM) circular convolution patches to achieve
    sun angle, shadow reversal, and sensor modality invariance.
    """

    def __init__(
        self,
        patch_size: int = 48,
        spatial_bins: int = 4,
        num_orientations: int = 6,
    ) -> None:
        self.patch_size = patch_size
        self.spatial_bins = spatial_bins
        self.num_orientations = num_orientations
        self.bin_size = max(2, patch_size // spatial_bins)
        self.descriptor_dim = spatial_bins * spatial_bins * num_orientations

    def compute_descriptors(
        self,
        mim: np.ndarray,
        keypoints: List[Tuple[float, float]],
    ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        h, w = mim.shape
        half = self.patch_size // 2
        descriptors: List[np.ndarray] = []
        valid_kps: List[Tuple[float, float]] = []

        for (x, y) in keypoints:
            ix, iy = int(round(x)), int(round(y))
            if (ix - half < 0 or ix + half >= w or
                iy - half < 0 or iy + half >= h):
                continue

            patch = mim[iy - half : iy + half, ix - half : ix + half]
            desc = np.zeros((self.spatial_bins, self.spatial_bins, self.num_orientations), dtype=np.float32)

            for by in range(self.spatial_bins):
                for bx in range(self.spatial_bins):
                    cell = patch[
                        by * self.bin_size : (by + 1) * self.bin_size,
                        bx * self.bin_size : (bx + 1) * self.bin_size,
                    ]
                    hist, _ = np.histogram(cell, bins=self.num_orientations, range=(0, self.num_orientations))
                    desc[by, bx, :] = hist

            flat_desc = desc.flatten()
            norm = np.linalg.norm(flat_desc) + 1e-7
            flat_desc /= norm

            # Non-linear saturation clipping
            flat_desc = np.minimum(flat_desc, 0.2)
            flat_desc /= (np.linalg.norm(flat_desc) + 1e-7)

            descriptors.append(flat_desc)
            valid_kps.append((x, y))

        if len(descriptors) == 0:
            return np.empty((0, self.descriptor_dim), dtype=np.float32), []

        return np.array(descriptors, dtype=np.float32), valid_kps


class ClassicalRIFTMatcher:
    """
    Classical Phase-Congruency RIFT Feature Matching Engine.
    Detects keypoints on phase congruency structural moments and matches via
    mutual nearest-neighbor distance ratio tests.
    """

    def __init__(
        self,
        max_features: int = 1500,
        patch_size: int = 48,
        spatial_bins: int = 4,
        num_orientations: int = 6,
        ratio_threshold: float = 0.85,
        mutual_check: bool = True,
    ) -> None:
        self.max_features = max_features
        self.extractor = RIFTDescriptorExtractor(
            patch_size=patch_size,
            spatial_bins=spatial_bins,
            num_orientations=num_orientations,
        )
        self.ratio_thresh = ratio_threshold
        self.mutual_check = mutual_check

    def detect_features(self, max_moment: np.ndarray) -> List[Tuple[float, float]]:
        """Detects repeatable corner/junction points on Phase Congruency moment map."""
        m_norm = cv2.normalize(max_moment.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        corners = cv2.goodFeaturesToTrack(
            m_norm,
            maxCorners=self.max_features,
            qualityLevel=0.02,
            minDistance=6.0,
            blockSize=5,
        )
        if corners is None or len(corners) == 0:
            return []
        return [(float(pt[0][0]), float(pt[0][1])) for pt in corners]

    def match(
        self,
        ref_max_moment: np.ndarray,
        ref_mim: np.ndarray,
        tgt_max_moment: np.ndarray,
        tgt_mim: np.ndarray,
    ) -> List[KeypointMatch]:
        """
        Runs classical RIFT feature detection and matching between reference and target.
        """
        kps_ref = self.detect_features(ref_max_moment)
        kps_tgt = self.detect_features(tgt_max_moment)

        if len(kps_ref) < 4 or len(kps_tgt) < 4:
            logger.warning('Insufficient keypoints detected for RIFT matching.')
            return []

        desc_ref, valid_ref = self.extractor.compute_descriptors(ref_mim, kps_ref)
        desc_tgt, valid_tgt = self.extractor.compute_descriptors(tgt_mim, kps_tgt)

        if len(valid_ref) < 4 or len(valid_tgt) < 4:
            return []

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        knn_matches = bf.knnMatch(desc_ref, desc_tgt, k=2)

        matches_ref_to_tgt: List[Tuple[int, int, float]] = []
        for pair in knn_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio_thresh * n.distance:
                confidence = float(1.0 - (m.distance / (n.distance + 1e-6)))
                matches_ref_to_tgt.append((m.queryIdx, m.trainIdx, confidence))

        if not self.mutual_check:
            return [
                KeypointMatch(
                    ref_xy=valid_ref[q_idx],
                    target_xy=valid_tgt[t_idx],
                    confidence=conf,
                )
                for (q_idx, t_idx, conf) in matches_ref_to_tgt
            ]

        # Reverse check
        rev_knn = bf.knnMatch(desc_tgt, desc_ref, k=2)
        tgt_to_ref = {pair[0].queryIdx: pair[0].trainIdx for pair in rev_knn if len(pair) >= 1}

        mutual_matches: List[KeypointMatch] = []
        for q_idx, t_idx, conf in matches_ref_to_tgt:
            if tgt_to_ref.get(t_idx) == q_idx:
                mutual_matches.append(
                    KeypointMatch(
                        ref_xy=valid_ref[q_idx],
                        target_xy=valid_tgt[t_idx],
                        confidence=conf,
                    )
                )

        logger.info(f'RIFT matched {len(mutual_matches)} points with mutual consistency.')
        return mutual_matches
