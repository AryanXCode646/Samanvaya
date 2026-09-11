"""
Strategy Selection and Multi-Modal Matcher Dispatcher.
SIH PS 26166: Dynamic matcher selection across optical, multi-resolution,
and cross-illumination lunar observations with transparent provenance.
"""

from __future__ import annotations

from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from lunar_core.models import KeypointMatch, SensorModality
from lunar_core.alignment.dense_matcher import DenseLoFTRMatcher
from lunar_core.alignment.rift_matcher import ClassicalRIFTMatcher
from lunar_core.preprocessing.phase_congruency import PhaseCongruencyEngine

logger = logging.getLogger(__name__)


class MatchingStrategy(str, Enum):
    """Supported correspondence matching paradigms."""
    DENSE_LOFTR = "dense_loftr"
    CLASSICAL_RIFT = "classical_rift"
    SIFT_ROOTSIFT = "sift_rootsift"
    PHASE_CORRELATION = "phase_correlation"
    HYBRID_CASCADE = "hybrid_cascade"


class ClassicalSIFTMatcher:
    """
    Classical Scale-Invariant Feature Transform (SIFT / RootSIFT) Matcher.
    Applies L1-RootSIFT normalization and mutual Lowe's ratio test.
    Suitable for moderate illumination differences and same-sensor pairs.
    """

    def __init__(
        self,
        nfeatures: int = 2000,
        ratio_threshold: float = 0.75,
        use_rootsift: bool = True,
    ) -> None:
        self.nfeatures = nfeatures
        self.ratio_threshold = ratio_threshold
        self.use_rootsift = use_rootsift
        self._sift = cv2.SIFT_create(nfeatures=nfeatures)

    def match(
        self,
        source_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> List[KeypointMatch]:
        def _to_u8(img: np.ndarray) -> np.ndarray:
            fin = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            mn, mx = float(np.min(fin)), float(np.max(fin))
            if mx - mn < 1e-6:
                return np.zeros(fin.shape, dtype=np.uint8)
            return np.clip((fin - mn) / (mx - mn) * 255.0, 0, 255).astype(np.uint8)

        src_u8 = _to_u8(source_image)
        ref_u8 = _to_u8(reference_image)

        kp_src, desc_src = self._sift.detectAndCompute(src_u8, None)
        kp_ref, desc_ref = self._sift.detectAndCompute(ref_u8, None)

        if desc_src is None or desc_ref is None or len(kp_src) < 4 or len(kp_ref) < 4:
            return []

        # RootSIFT L1-sqrt normalization
        if self.use_rootsift:
            desc_src = desc_src / (np.sum(np.abs(desc_src), axis=1, keepdims=True) + 1e-7)
            desc_src = np.sqrt(np.clip(desc_src, 0.0, None))
            desc_ref = desc_ref / (np.sum(np.abs(desc_ref), axis=1, keepdims=True) + 1e-7)
            desc_ref = np.sqrt(np.clip(desc_ref, 0.0, None))

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        raw_matches = bf.knnMatch(desc_src, desc_ref, k=2)

        matches: List[KeypointMatch] = []
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            m, n = pair[0], pair[1]
            if m.distance < self.ratio_threshold * n.distance:
                pt_src = kp_src[m.queryIdx].pt
                pt_ref = kp_ref[m.trainIdx].pt
                conf = float(1.0 - (m.distance / (n.distance + 1e-7)))
                matches.append(
                    KeypointMatch(
                        ref_xy=(float(pt_ref[0]), float(pt_ref[1])),
                        target_xy=(float(pt_src[0]), float(pt_src[1])),
                        confidence=max(0.1, min(1.0, conf)),
                        source_frame="FULL_SOURCE_IMAGE",
                        reference_frame="FULL_REFERENCE_IMAGE",
                    )
                )

        return matches


class PhaseCorrelationMatcher:
    """
    Fourier-Mellin phase correlation for translation and rotation pre-alignment.
    Operates in the frequency domain; highly robust to uniform illumination changes.
    """

    @staticmethod
    def estimate_translation(
        source_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> Tuple[Tuple[float, float], float]:
        """Estimate global (dx, dy) translation via phase correlation."""
        h1, w1 = source_image.shape[:2]
        h2, w2 = reference_image.shape[:2]
        target_h, target_w = min(h1, h2), min(w1, w2)
        s_crop = cv2.resize(source_image[:target_h, :target_w].astype(np.float32), (target_w, target_h))
        r_crop = cv2.resize(reference_image[:target_h, :target_w].astype(np.float32), (target_w, target_h))

        # Hann windowing to suppress edge discontinuities
        window = cv2.createHanningWindow((target_w, target_h), cv2.CV_32F)
        shift, response = cv2.phaseCorrelate(s_crop * window, r_crop * window)
        return (float(shift[0]), float(shift[1])), float(response)


class MatchingStrategySelector:
    """
    Intelligent Strategy Selection Layer.
    Evaluates mission product metadata, solar angles, GSD ratio, and model weights
    to select and execute the most scientifically defensible matcher.
    """

    def __init__(
        self,
        preferred_strategy: Optional[MatchingStrategy] = None,
        pretrained_weights: Optional[str] = "outdoor",
        confidence_threshold: float = 0.15,
        device: Optional[str] = None,
    ) -> None:
        self.preferred_strategy = preferred_strategy
        self.pretrained_weights = pretrained_weights
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.pc_engine = PhaseCongruencyEngine()

    def select_strategy(
        self,
        source_modality: Optional[SensorModality | str] = None,
        reference_modality: Optional[SensorModality | str] = None,
        gsd_ratio: Optional[float] = None,
        illumination_delta_deg: Optional[float] = None,
        has_learned_weights: bool = True,
    ) -> Tuple[MatchingStrategy, str]:
        """
        Determines the optimal matching strategy based on physical and operational constraints.
        Returns (strategy, reason).
        """
        if self.preferred_strategy is not None:
            return self.preferred_strategy, f"Explicitly configured: {self.preferred_strategy.value}"

        # 1. Hyperspectral or multi-band cross-modal
        is_spectral = str(source_modality).upper() in {"IIRS", "HYSI"} or str(reference_modality).upper() in {"IIRS", "HYSI"}
        if is_spectral:
            return MatchingStrategy.CLASSICAL_RIFT, "Hyperspectral cross-modal: Classical RIFT on phase congruency moments is most invariant."

        # 2. Large illumination differences (> 45 deg)
        if illumination_delta_deg is not None and illumination_delta_deg > 45.0:
            if has_learned_weights:
                return MatchingStrategy.DENSE_LOFTR, f"Severe illumination disparity ({illumination_delta_deg:.1f}°): Dense LoFTR with phase congruency."
            return MatchingStrategy.CLASSICAL_RIFT, f"Severe illumination disparity ({illumination_delta_deg:.1f}°) with no learned weights: Classical RIFT."

        # 3. High resolution disparity (GSD ratio > 4.0)
        if gsd_ratio is not None and gsd_ratio > 4.0:
            return MatchingStrategy.HYBRID_CASCADE, f"Multi-scale disparity (GSD ratio={gsd_ratio:.1f}x): Coarse-to-fine hybrid cascade."

        # 4. Same modality optical with learned weights
        if has_learned_weights:
            return MatchingStrategy.DENSE_LOFTR, "Default optical registration: Dense LoFTR cross-attention matching."

        # 5. Fallback when learned weights are absent
        return MatchingStrategy.CLASSICAL_RIFT, "Offline CPU/No learned weights: Classical RIFT on invariant structural moments."

    def match(
        self,
        source_image: np.ndarray,
        reference_image: np.ndarray,
        source_modality: Optional[SensorModality | str] = None,
        reference_modality: Optional[SensorModality | str] = None,
        gsd_ratio: Optional[float] = None,
        illumination_delta_deg: Optional[float] = None,
    ) -> Tuple[List[KeypointMatch], Dict[str, Any]]:
        """
        Executes correspondence matching via selected strategy with automatic fallback.
        """
        loftr = DenseLoFTRMatcher(
            pretrained=self.pretrained_weights,
            confidence_threshold=self.confidence_threshold,
            device=self.device,
        )

        strategy, reason = self.select_strategy(
            source_modality=source_modality,
            reference_modality=reference_modality,
            gsd_ratio=gsd_ratio,
            illumination_delta_deg=illumination_delta_deg,
            has_learned_weights=loftr.is_pretrained,
        )

        provenance: Dict[str, Any] = {
            "selected_strategy": strategy.value,
            "strategy_reason": reason,
            "learned_weights_used": loftr.is_pretrained,
            "weight_source": loftr.weight_source,
            "fallback_chain": [],
        }

        matches: List[KeypointMatch] = []

        # Strategy Execution
        if strategy in {MatchingStrategy.DENSE_LOFTR, MatchingStrategy.HYBRID_CASCADE}:
            try:
                # Dense LoFTR path
                src_tensor, norm_src = loftr.prepare_geotiff_array(source_image)
                ref_tensor, norm_ref = loftr.prepare_geotiff_array(reference_image)
                matches = loftr.extract_dense_correspondences(
                    src_tensor, ref_tensor, norm_src.shape, norm_ref.shape
                )
            except Exception as exc:
                logger.warning("LoFTR execution failed (%s); triggering Classical RIFT fallback.", exc)
                provenance["fallback_chain"].append({"from": strategy.value, "to": "classical_rift", "reason": str(exc)})
                strategy = MatchingStrategy.CLASSICAL_RIFT

        if strategy == MatchingStrategy.CLASSICAL_RIFT or (not matches and len(provenance["fallback_chain"]) > 0):
            try:
                pc_src = self.pc_engine.compute(source_image)
                pc_ref = self.pc_engine.compute(reference_image)
                rift = ClassicalRIFTMatcher()
                matches = rift.match(
                    pc_ref.max_moment,
                    pc_ref.orientation_max_idx,
                    pc_src.max_moment,
                    pc_src.orientation_max_idx,
                )
                provenance["active_strategy"] = MatchingStrategy.CLASSICAL_RIFT.value
            except Exception as exc:
                logger.warning("Classical RIFT failed (%s); trying RootSIFT fallback.", exc)
                provenance["fallback_chain"].append({"from": "classical_rift", "to": "sift_rootsift", "reason": str(exc)})
                strategy = MatchingStrategy.SIFT_ROOTSIFT

        if strategy == MatchingStrategy.SIFT_ROOTSIFT or not matches:
            try:
                sift = ClassicalSIFTMatcher()
                matches = sift.match(source_image, reference_image)
                provenance["active_strategy"] = MatchingStrategy.SIFT_ROOTSIFT.value
            except Exception as exc:
                logger.error("All correspondence matchers failed: %s", exc)
                provenance["fallback_chain"].append({"from": "sift_rootsift", "to": "failed", "reason": str(exc)})

        provenance["raw_match_count"] = len(matches)
        return matches, provenance
