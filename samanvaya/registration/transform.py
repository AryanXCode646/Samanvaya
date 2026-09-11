"""Typed source-to-reference registration transforms and geometric validation gates."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import cv2
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


def select_geometric_model(
    matches: list[Any],
    reproj_threshold: float = 1.5,
    min_inliers: int = 4,
    preferred_model: Optional[str] = None,
) -> Tuple[Optional[RegistrationTransform], Dict[str, Any], list[Any]]:
    """
    Evaluates candidate geometric transformation models (TRANSLATION, SIMILARITY, AFFINE, HOMOGRAPHY)
    based on inlier consensus, Bayesian Information Criterion (BIC), condition number, and degeneracy checks.
    Rejects degenerate transforms.

    Returns:
        (best_transform, diagnostics_report, inlier_matches)
    """
    if len(matches) < min_inliers:
        return None, {
            "status": "INSUFFICIENT_MATCHES",
            "reason": f"At least {min_inliers} matches required, got {len(matches)}.",
            "candidates": {},
        }, []

    src_pts = np.array([m.target_xy for m in matches], dtype=np.float64)
    ref_pts = np.array([m.ref_xy for m in matches], dtype=np.float64)
    n = len(matches)

    candidates: Dict[str, Dict[str, Any]] = {}

    # Candidate 1: TRANSLATION (2 DOF: dx, dy)
    dx = float(np.median(ref_pts[:, 0] - src_pts[:, 0]))
    dy = float(np.median(ref_pts[:, 1] - src_pts[:, 1]))
    t_mat = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float64)
    pred_t = src_pts + np.array([dx, dy])
    res_t = np.linalg.norm(ref_pts - pred_t, axis=1)
    inliers_t = np.where(res_t <= reproj_threshold)[0]
    rmse_t = float(np.sqrt(np.mean(res_t[inliers_t] ** 2))) if len(inliers_t) >= min_inliers else 999.0
    bic_t = float(len(inliers_t) * np.log(rmse_t**2 + 1e-6) + 2.0 * np.log(max(len(inliers_t), 1)))
    candidates["TRANSLATION"] = {
        "matrix": t_mat,
        "dof": 2,
        "inlier_count": len(inliers_t),
        "inlier_ratio": float(len(inliers_t) / n),
        "rmse": rmse_t,
        "bic": bic_t,
        "residuals": res_t,
        "inlier_indices": inliers_t,
    }

    # Candidate 2: SIMILARITY (4 DOF: scale, rotation, dx, dy)
    try:
        sim_mat_2x3, mask_sim = cv2.estimateAffinePartial2D(
            src_pts.astype(np.float32),
            ref_pts.astype(np.float32),
            method=getattr(cv2, "USAC_MAGSAC", cv2.RANSAC),
            ransacReprojThreshold=reproj_threshold,
            maxIters=5000,
            confidence=0.999,
        )
        if sim_mat_2x3 is not None and mask_sim is not None:
            s_mat = np.vstack([sim_mat_2x3, [0.0, 0.0, 1.0]]).astype(np.float64)
            inliers_s = np.where(mask_sim.ravel() == 1)[0]
            src_h = np.column_stack([src_pts, np.ones(n)])
            pred_s = (src_h @ s_mat.T)[:, :2]
            res_s = np.linalg.norm(ref_pts - pred_s, axis=1)
            rmse_s = float(np.sqrt(np.mean(res_s[inliers_s] ** 2))) if len(inliers_s) >= min_inliers else 999.0
            bic_s = float(len(inliers_s) * np.log(rmse_s**2 + 1e-6) + 4.0 * np.log(max(len(inliers_s), 1)))
            candidates["SIMILARITY"] = {
                "matrix": s_mat,
                "dof": 4,
                "inlier_count": len(inliers_s),
                "inlier_ratio": float(len(inliers_s) / n),
                "rmse": rmse_s,
                "bic": bic_s,
                "residuals": res_s,
                "inlier_indices": inliers_s,
            }
    except Exception:
        pass

    # Candidate 3: AFFINE (6 DOF)
    try:
        aff_mat_2x3, mask_aff = cv2.estimateAffine2D(
            src_pts.astype(np.float32),
            ref_pts.astype(np.float32),
            method=getattr(cv2, "USAC_MAGSAC", cv2.RANSAC),
            ransacReprojThreshold=reproj_threshold,
            maxIters=5000,
            confidence=0.999,
        )
        if aff_mat_2x3 is not None and mask_aff is not None:
            a_mat = np.vstack([aff_mat_2x3, [0.0, 0.0, 1.0]]).astype(np.float64)
            inliers_a = np.where(mask_aff.ravel() == 1)[0]
            src_h = np.column_stack([src_pts, np.ones(n)])
            pred_a = (src_h @ a_mat.T)[:, :2]
            res_a = np.linalg.norm(ref_pts - pred_a, axis=1)
            rmse_a = float(np.sqrt(np.mean(res_a[inliers_a] ** 2))) if len(inliers_a) >= min_inliers else 999.0
            bic_a = float(len(inliers_a) * np.log(rmse_a**2 + 1e-6) + 6.0 * np.log(max(len(inliers_a), 1)))
            candidates["AFFINE"] = {
                "matrix": a_mat,
                "dof": 6,
                "inlier_count": len(inliers_a),
                "inlier_ratio": float(len(inliers_a) / n),
                "rmse": rmse_a,
                "bic": bic_a,
                "residuals": res_a,
                "inlier_indices": inliers_a,
            }
    except Exception:
        pass

    # Candidate 4: HOMOGRAPHY (8 DOF)
    try:
        h_mat, mask_h = cv2.findHomography(
            src_pts.astype(np.float32),
            ref_pts.astype(np.float32),
            method=getattr(cv2, "USAC_MAGSAC", cv2.RANSAC),
            ransacReprojThreshold=reproj_threshold,
            maxIters=5000,
            confidence=0.999,
        )
        if h_mat is not None and mask_h is not None:
            h_mat = h_mat.astype(np.float64)
            inliers_h = np.where(mask_h.ravel() == 1)[0]
            src_h = np.column_stack([src_pts, np.ones(n)])
            proj_h = src_h @ h_mat.T
            denom = np.where(np.abs(proj_h[:, 2:3]) < 1e-12, 1e-12, proj_h[:, 2:3])
            pred_h = proj_h[:, :2] / denom
            res_h = np.linalg.norm(ref_pts - pred_h, axis=1)
            rmse_h = float(np.sqrt(np.mean(res_h[inliers_h] ** 2))) if len(inliers_h) >= min_inliers else 999.0
            bic_h = float(len(inliers_h) * np.log(rmse_h**2 + 1e-6) + 8.0 * np.log(max(len(inliers_h), 1)))
            candidates["HOMOGRAPHY"] = {
                "matrix": h_mat,
                "dof": 8,
                "inlier_count": len(inliers_h),
                "inlier_ratio": float(len(inliers_h) / n),
                "rmse": rmse_h,
                "bic": bic_h,
                "residuals": res_h,
                "inlier_indices": inliers_h,
            }
    except Exception:
        pass

    # Model selection
    valid_candidates = {
        name: data
        for name, data in candidates.items()
        if data["inlier_count"] >= min_inliers and np.all(np.isfinite(data["matrix"]))
    }

    if not valid_candidates:
        return None, {
            "status": "DEGENERATE_TRANSFORM",
            "reason": "All transformation models failed to achieve minimum inlier count.",
            "candidates": {k: {"inliers": v["inlier_count"], "rmse": v["rmse"]} for k, v in candidates.items()},
        }, []

    # If preferred_model is specified and valid, use it
    if preferred_model and preferred_model.upper() in valid_candidates:
        best_name = preferred_model.upper()
    else:
        # Sort primarily by inlier count, then by lowest BIC score
        best_name = max(
            valid_candidates.keys(),
            key=lambda k: (valid_candidates[k]["inlier_count"], -valid_candidates[k]["bic"]),
        )

    best = valid_candidates[best_name]
    mat = best["matrix"]
    reg_tf = RegistrationTransform(
        model_type=best_name,
        source_frame="FULL_SOURCE_IMAGE",
        target_frame="FULL_REFERENCE_IMAGE",
        parameters=mat if mat.shape == (3, 3) else np.vstack([mat, [0.0, 0.0, 1.0]]),
        estimation_method=f"USAC_MAGSAC_BIC_SELECTION({best_name})",
        fit_statistics={
            "dof": best["dof"],
            "inlier_count": best["inlier_count"],
            "inlier_ratio": best["inlier_ratio"],
            "rmse": best["rmse"],
            "bic": best["bic"],
            "condition_number": float(np.linalg.cond(mat)),
        },
    )

    is_plausible, plaus_reason = check_geometric_plausibility(reg_tf)
    if not is_plausible:
        return None, {
            "status": "IMPLAUSIBLE_TRANSFORM",
            "reason": plaus_reason,
            "selected_model": best_name,
            "condition_number": reg_tf.condition_number(),
            "candidates": {k: {"inliers": v["inlier_count"], "rmse": v["rmse"]} for k, v in candidates.items()},
        }, []

    inlier_indices = best["inlier_indices"]
    inlier_matches = [matches[idx] for idx in inlier_indices]

    diagnostics = {
        "status": "SUCCESS",
        "selected_model": best_name,
        "parameter_estimates": mat.tolist(),
        "number_of_matches": n,
        "number_of_inliers": best["inlier_count"],
        "inlier_ratio": best["inlier_ratio"],
        "reprojection_consensus_error": best["rmse"],
        "condition_number": reg_tf.condition_number(),
        "is_plausible": is_plausible,
        "candidate_comparison": {
            k: {
                "inlier_count": v["inlier_count"],
                "inlier_ratio": round(v["inlier_ratio"], 4),
                "reprojection_rmse": round(v["rmse"], 4),
                "bic": round(v["bic"], 2),
            }
            for k, v in candidates.items()
        },
    }

    return reg_tf, diagnostics, inlier_matches
