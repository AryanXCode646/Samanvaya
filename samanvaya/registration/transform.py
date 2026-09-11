"""Typed source-to-reference registration transforms and geometric validation gates."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class RegistrationTransform:
    """Authoritative convention: source FULL_IMAGE -> reference FULL_IMAGE."""

    model_type: str
    source_frame: str
    target_frame: str
    parameters: np.ndarray
    estimation_method: str
    fit_statistics: Dict[str, Any] = field(default_factory=dict)
    coordinate_convention: str = "pixel centers, x=column, y=row; source -> reference"
    direction: str = "source->reference"

    @property
    def inverse_available(self) -> bool:
        if self.parameters is None:
            return False
        h = self.as_homography()
        if not np.all(np.isfinite(h)):
            return False
        det = np.linalg.det(h)
        return abs(det) > 1e-12 and np.linalg.matrix_rank(h) == 3

    def as_homography(self) -> np.ndarray:
        if self.parameters.shape == (2, 3):
            return np.vstack([self.parameters, [0.0, 0.0, 1.0]]).astype(np.float64)
        if self.parameters.shape == (3, 3):
            return self.parameters.astype(np.float64)
        raise ValueError(f"Unsupported transform shape: {self.parameters.shape}")

    def apply_source_to_reference(
        self,
        points: np.ndarray,
        source_frame: Optional[str] = None,
    ) -> np.ndarray:
        if source_frame is not None and source_frame != self.source_frame:
            raise ValueError("MIXED_COORDINATE_FRAMES")
        return _apply(points, self.as_homography())

    def apply_reference_to_source(
        self,
        points: np.ndarray,
        target_frame: Optional[str] = None,
    ) -> np.ndarray:
        if target_frame is not None and target_frame != self.target_frame:
            raise ValueError("MIXED_COORDINATE_FRAMES")
        if not self.inverse_available:
            raise ValueError("DEGENERATE_TRANSFORM: inverse is unavailable")
        return _apply(points, np.linalg.inv(self.as_homography()))

    def inverse(self) -> "RegistrationTransform":
        if not self.inverse_available:
            raise ValueError("DEGENERATE_TRANSFORM: inverse is unavailable")
        h_inv = np.linalg.inv(self.as_homography())
        inv_params = h_inv if self.parameters.shape == (3, 3) else h_inv[:2, :]
        return RegistrationTransform(
            model_type=self.model_type,
            source_frame=self.target_frame,
            target_frame=self.source_frame,
            parameters=inv_params,
            estimation_method=f"inverse({self.estimation_method})",
            fit_statistics=dict(self.fit_statistics),
            coordinate_convention=self.coordinate_convention,
            direction="reference->source",
        )

    def condition_number(self) -> float:
        try:
            return float(np.linalg.cond(self.as_homography()))
        except Exception:
            return float("inf")


def _apply(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if len(values) == 0:
        return np.empty((0, 2), dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, 2)
    homogeneous = np.column_stack([values, np.ones(len(values), dtype=np.float64)])
    projected = homogeneous @ matrix.T
    denom = projected[:, 2:3]
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    return projected[:, :2] / denom


def check_geometric_plausibility(
    transform: RegistrationTransform,
    scale_range: Tuple[float, float] = (0.05, 20.0),
    max_rotation_deg: float = 360.0,
    max_shear: float = 2.0,
) -> Tuple[bool, str]:
    """
    Validates transform against physical remote-sensing sanity ranges.
    Returns (is_plausible, reason).
    """
    h = transform.as_homography()
    if not np.all(np.isfinite(h)):
        return False, "DEGENERATE_TRANSFORM: matrix contains NaN or Inf"

    det = float(np.linalg.det(h))
    if det <= 1e-7:
        return False, "IMPLAUSIBLE_TRANSFORM: negative or non-positive determinant"

    # Inspect 2x2 linear component for affine / first-order projective behavior
    a11, a12 = h[0, 0] / (h[2, 2] + 1e-12), h[0, 1] / (h[2, 2] + 1e-12)
    a21, a22 = h[1, 0] / (h[2, 2] + 1e-12), h[1, 1] / (h[2, 2] + 1e-12)
    linear = np.array([[a11, a12], [a21, a22]], dtype=np.float64)

    # Singular values give scale factors along major axes
    s = np.linalg.svd(linear, compute_uv=False)
    s_min, s_max = float(s[1]), float(s[0])

    if s_min < scale_range[0] or s_max > scale_range[1]:
        return False, f"IMPLAUSIBLE_TRANSFORM: scale {s_min:.3f}..{s_max:.3f} outside bounds {scale_range}"

    # Aspect ratio / shear check
    aspect_ratio = s_max / (s_min + 1e-12)
    if aspect_ratio > (1.0 + max_shear):
        return False, f"IMPLAUSIBLE_TRANSFORM: excessive shear/aspect ratio {aspect_ratio:.2f}"

    # Projective distortion terms check
    h31, h32 = abs(h[2, 0] / (h[2, 2] + 1e-12)), abs(h[2, 1] / (h[2, 2] + 1e-12))
    if h31 > 0.05 or h32 > 0.05:
        return False, f"IMPLAUSIBLE_TRANSFORM: excessive projective tilt ({h31:.4f}, {h32:.4f})"

    return True, "PLAUSIBLE"


def check_spatial_model_mismatch(
    reference_pts: np.ndarray,
    residuals: np.ndarray,
    correlation_threshold: float = 0.65,
) -> Tuple[bool, str]:
    """
    Evaluates whether residuals exhibit strong systematic spatial structure,
    indicating that a single global planar model is physically insufficient.
    """
    ref = np.asarray(reference_pts, dtype=np.float64)
    res = np.asarray(residuals, dtype=np.float64)
    if len(ref) < 8 or len(res) < 8:
        return True, "INSUFFICIENT_POINTS_FOR_SPATIAL_AUDIT"

    # Check correlation between spatial coordinates (x, y) and residual error magnitude
    rx = ref[:, 0] - np.mean(ref[:, 0])
    ry = ref[:, 1] - np.mean(ref[:, 1])
    std_res = np.std(res)
    if std_res < 1e-6:
        return True, "UNIFORM_RESIDUALS"

    r_res = (res - np.mean(res)) / std_res

    corr_x = abs(float(np.mean((rx / (np.std(rx) + 1e-8)) * r_res)))
    corr_y = abs(float(np.mean((ry / (np.std(ry) + 1e-8)) * r_res)))

    if corr_x > correlation_threshold or corr_y > correlation_threshold:
        return False, f"MODEL_MISMATCH: systematic spatial residual gradient (corr_x={corr_x:.2f}, corr_y={corr_y:.2f})"

    return True, "GLOBAL_MODEL_ACCEPTED"
