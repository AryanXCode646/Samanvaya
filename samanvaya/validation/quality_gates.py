"""
Pre-Registration Quality Gates (SIH PS 26166).

Defensive pre-flight validation to prevent ill-conditioned or mathematically
impossible registration attempts on corrupt, textureless, or non-overlapping images.

Guarantees explicit, informative rejection codes rather than silent degradation:
  - DATA_REQUIRED: Missing essential image data or empty arrays
  - NON_OVERLAPPING: Bounding polygons or footprints share 0% overlap
  - LOW_TEXTURE: Image lacks high-frequency gradient entropy (e.g. blank, flat sky, saturated whiteout)
  - INSUFFICIENT_VALID_AREA: Valid pixel fraction below operational threshold (e.g. nodata/nans > 85%)
  - EXCESSIVE_SHADOW_SATURATION: Dynamic range collapsed due to deep lunar shadowing or sensor saturation
  - INVALID_GEOMETRY: Degenerate raster dimensions (< 16x16) or corrupt shape
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class QualityGateResult:
    """
    Structured outcome of pre-registration quality evaluation.
    """
    passed: bool
    rejection_code: Optional[str]  # e.g., DATA_REQUIRED, NON_OVERLAPPING, LOW_TEXTURE, etc.
    metrics: Dict[str, float] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "rejection_code": self.rejection_code,
            "metrics": self.metrics,
            "message": self.message,
        }


def compute_image_quality_metrics(image: np.ndarray) -> Dict[str, float]:
    """
    Compute defensive radiometric and texture metrics for a 2D single-band image.
    """
    if image is None or image.size == 0:
        return {
            "valid_fraction": 0.0,
            "shadow_saturation_fraction": 1.0,
            "normalized_variance": 0.0,
            "gradient_energy": 0.0,
        }

    # Ensure 2D
    img = np.squeeze(image).astype(np.float64)
    if img.ndim != 2:
        img = img[..., 0] if img.ndim > 2 else img.flatten()

    total_pixels = img.size
    finite_mask = np.isfinite(img)
    valid_pixels = np.count_nonzero(finite_mask)
    if valid_pixels == 0:
        return {
            "valid_fraction": 0.0,
            "shadow_saturation_fraction": 1.0,
            "normalized_variance": 0.0,
            "gradient_energy": 0.0,
        }

    valid_vals = img[finite_mask]
    valid_fraction = float(valid_pixels / total_pixels)

    min_v = float(np.min(valid_vals))
    max_v = float(np.max(valid_vals))
    span = max_v - min_v

    if span <= 1e-12:
        # Flat image
        return {
            "valid_fraction": valid_fraction,
            "shadow_saturation_fraction": 1.0,
            "normalized_variance": 0.0,
            "gradient_energy": 0.0,
        }

    # Normalize valid values to [0, 1]
    norm_vals = (valid_vals - min_v) / span
    norm_var = float(np.var(norm_vals))

    # Shadow & saturation check: pixels in lowest 2% or highest 2% of range
    shadow_sat_count = np.count_nonzero((norm_vals <= 0.02) | (norm_vals >= 0.98))
    shadow_sat_frac = float(shadow_sat_count / valid_pixels)

    # Gradient energy approximation via 2D spatial finite differences
    # Compute on finite image with NaNs replaced by median
    filled_img = np.where(finite_mask, img, np.median(valid_vals))
    gy, gx = np.gradient(filled_img)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    grad_energy = float(np.mean(grad_mag[finite_mask])) / (span + 1e-8)

    return {
        "valid_fraction": valid_fraction,
        "shadow_saturation_fraction": shadow_sat_frac,
        "normalized_variance": norm_var,
        "gradient_energy": grad_energy,
    }


def evaluate_pre_registration_gates(
    source_image: Optional[np.ndarray],
    ref_image: Optional[np.ndarray],
    overlap_ratio: Optional[float] = None,
    min_valid_fraction: float = 0.15,
    max_shadow_saturation_fraction: float = 0.95,
    min_normalized_variance: float = 1e-4,
    min_dim: int = 16,
) -> QualityGateResult:
    """
    Strict quality gating before feature extraction and transformation solving.
    """
    # 1. Null / emptiness gate
    if source_image is None or ref_image is None:
        return QualityGateResult(
            passed=False,
            rejection_code="DATA_REQUIRED",
            message="Source or reference image array is None or missing.",
        )

    if source_image.size == 0 or ref_image.size == 0:
        return QualityGateResult(
            passed=False,
            rejection_code="DATA_REQUIRED",
            message="Source or reference image array is empty (0 pixels).",
        )

    # 2. Geometric dimension gate
    h_src, w_src = source_image.shape[:2]
    h_ref, w_ref = ref_image.shape[:2]
    if h_src < min_dim or w_src < min_dim or h_ref < min_dim or w_ref < min_dim:
        return QualityGateResult(
            passed=False,
            rejection_code="INVALID_GEOMETRY",
            metrics={"src_h": float(h_src), "src_w": float(w_src), "ref_h": float(h_ref), "ref_w": float(w_ref)},
            message=f"Raster dimensions ({w_src}x{h_src} or {w_ref}x{h_ref}) below required minimum {min_dim}px.",
        )

    # 3. Explicit overlap gate
    if overlap_ratio is not None:
        if overlap_ratio <= 0.0:
            return QualityGateResult(
                passed=False,
                rejection_code="NON_OVERLAPPING",
                metrics={"overlap_ratio": float(overlap_ratio)},
                message=f"Zero geospatial overlap detected between footprints (overlap_ratio = {overlap_ratio:.4f}).",
            )
        if overlap_ratio < 0.05:
            return QualityGateResult(
                passed=False,
                rejection_code="NON_OVERLAPPING",
                metrics={"overlap_ratio": float(overlap_ratio)},
                message=f"Geospatial overlap ({overlap_ratio:.2%}) below minimum viable threshold 5.00%.",
            )

    # 4. Radiometric and texture metrics
    src_metrics = compute_image_quality_metrics(source_image)
    ref_metrics = compute_image_quality_metrics(ref_image)

    combined_metrics = {
        "src_valid_fraction": src_metrics["valid_fraction"],
        "ref_valid_fraction": ref_metrics["valid_fraction"],
        "src_shadow_sat_fraction": src_metrics["shadow_saturation_fraction"],
        "ref_shadow_sat_fraction": ref_metrics["shadow_saturation_fraction"],
        "src_variance": src_metrics["normalized_variance"],
        "ref_variance": ref_metrics["normalized_variance"],
        "src_gradient_energy": src_metrics["gradient_energy"],
        "ref_gradient_energy": ref_metrics["gradient_energy"],
    }
    if overlap_ratio is not None:
        combined_metrics["overlap_ratio"] = float(overlap_ratio)

    # Valid area check
    if (src_metrics["valid_fraction"] < min_valid_fraction or 
            ref_metrics["valid_fraction"] < min_valid_fraction):
        min_f = min(src_metrics["valid_fraction"], ref_metrics["valid_fraction"])
        return QualityGateResult(
            passed=False,
            rejection_code="INSUFFICIENT_VALID_AREA",
            metrics=combined_metrics,
            message=f"Valid pixel fraction ({min_f:.2%}) below operational floor {min_valid_fraction:.2%}.",
        )

    # Flat / zero texture check
    if (src_metrics["normalized_variance"] < min_normalized_variance or 
            ref_metrics["normalized_variance"] < min_normalized_variance):
        min_v = min(src_metrics["normalized_variance"], ref_metrics["normalized_variance"])
        return QualityGateResult(
            passed=False,
            rejection_code="LOW_TEXTURE",
            metrics=combined_metrics,
            message=f"Normalized texture variance ({min_v:.2e}) insufficient for reliable feature correspondence.",
        )

    # Excessive shadow / saturation check
    if (src_metrics["shadow_saturation_fraction"] > max_shadow_saturation_fraction or 
            ref_metrics["shadow_saturation_fraction"] > max_shadow_saturation_fraction):
        max_s = max(src_metrics["shadow_saturation_fraction"], ref_metrics["shadow_saturation_fraction"])
        return QualityGateResult(
            passed=False,
            rejection_code="EXCESSIVE_SHADOW_SATURATION",
            metrics=combined_metrics,
            message=f"Extreme dynamic range collapse: {max_s:.2%} of pixels in extreme shadow/saturation.",
        )

    return QualityGateResult(
        passed=True,
        rejection_code=None,
        metrics=combined_metrics,
        message="All pre-registration quality gates passed successfully.",
    )
