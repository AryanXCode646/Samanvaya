"""
Sub-Pixel Peak Estimator via Analytical 2D Bivariate Quadratic Patches.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np

from lunar_core.models import KeypointMatch


@dataclass
class SubpixelSurfaceFit:
    """
    Result of continuous 2D quadratic patch fitting.
    Supports backward-compatible tuple unpacking: dx, dy, peak_val = fit
    """
    dx: float
    dy: float
    peak_val: float
    sigma_x: float
    sigma_y: float
    cov_xy: float
    weight: float
    eigenvalues: Tuple[float, float] = (0.0, 0.0)

    def __iter__(self):
        yield self.dx
        yield self.dy
        yield self.peak_val

    def as_tuple(self) -> Tuple[float, float, float, float, float, float, float]:
        return (
            self.dx,
            self.dy,
            self.peak_val,
            self.sigma_x,
            self.sigma_y,
            self.cov_xy,
            self.weight,
        )


def fit_quadratic_peak(patch_3x3: np.ndarray) -> Optional[SubpixelSurfaceFit]:
    r"""
    Fits an analytical 2D bivariate quadratic surface:
        f(x, y) = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f
    around integer grid point (0, 0) over a 3x3 local neighborhood.
    """
    if patch_3x3.shape != (3, 3):
        raise ValueError(f"Expected 3x3 patch, got shape {patch_3x3.shape}")

    z00 = float(patch_3x3[1, 1])
    z_xp = float(patch_3x3[1, 2])
    z_xm = float(patch_3x3[1, 0])
    z_yp = float(patch_3x3[2, 1])
    z_ym = float(patch_3x3[0, 1])
    z_pp = float(patch_3x3[2, 2])
    z_pm = float(patch_3x3[0, 2])
    z_mp = float(patch_3x3[2, 0])
    z_mm = float(patch_3x3[0, 0])

    a = (z_xp - 2.0 * z00 + z_xm) / 2.0
    b = (z_yp - 2.0 * z00 + z_ym) / 2.0
    c = (z_pp - z_pm - z_mp + z_mm) / 4.0
    d = (z_xp - z_xm) / 2.0
    e = (z_yp - z_ym) / 2.0
    f = z00

    det_h = 4.0 * a * b - c**2
    if det_h <= 1e-7 or a >= 0.0 or b >= 0.0:
        return None

    dx = (-2.0 * b * d + c * e) / det_h
    dy = (-2.0 * a * e + c * d) / det_h

    if abs(dx) > 1.0 or abs(dy) > 1.0:
        return None

    peak_val = a * dx**2 + b * dy**2 + c * dx * dy + d * dx + e * dy + f

    sigma_xx = (-2.0 * b) / det_h
    sigma_yy = (-2.0 * a) / det_h
    cov_xy = c / det_h

    sigma_x = float(np.sqrt(max(sigma_xx, 0.0)))
    sigma_y = float(np.sqrt(max(sigma_yy, 0.0)))

    trace_cov = sigma_xx + sigma_yy
    det_cov = 1.0 / det_h
    discriminant = max(0.0, trace_cov**2 - 4.0 * det_cov)
    sqrt_disc = float(np.sqrt(discriminant))

    lambda_1 = float(0.5 * (trace_cov + sqrt_disc))
    lambda_2 = float(0.5 * (trace_cov - sqrt_disc))
    weight = float(np.sqrt(det_h))

    return SubpixelSurfaceFit(
        dx=float(dx),
        dy=float(dy),
        peak_val=float(peak_val),
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        cov_xy=float(cov_xy),
        weight=weight,
        eigenvalues=(lambda_1, lambda_2),
    )


def fit_quadratic_peaks_batch(patches_3x3: np.ndarray) -> List[Optional[SubpixelSurfaceFit]]:
    """
    Batch quadratic surface fitting for an array of N 3x3 patches.
    """
    patches = np.asarray(patches_3x3, dtype=np.float64)
    if patches.ndim != 3 or patches.shape[1:] != (3, 3):
        raise ValueError(f"Expected array of shape (N, 3, 3), got {patches.shape}")
    return [fit_quadratic_peak(p) for p in patches]


class SubpixelRefinerBase(ABC):
    """
    Abstract Base Class for sub-pixel keypoint refinement.
    Defines unified contract for continuous surface fitting and batch refinement.
    """

    @abstractmethod
    def fit_surface(self, patch_3x3: np.ndarray) -> Optional[SubpixelSurfaceFit]:
        """Fits continuous bivariate surface over a 3x3 similarity patch."""
        pass

    @abstractmethod
    def refine_matches_batch(
        self,
        matches: List[KeypointMatch],
        ref_moment: np.ndarray,
        tgt_moment: np.ndarray,
        patch_radius: int = 8,
    ) -> List[KeypointMatch]:
        """Refines a batch of keypoint correspondences to sub-pixel accuracy."""
        pass


class ParabolicHessianRefiner(SubpixelRefinerBase):
    r"""
    Fits an analytical 2D quadratic patch:
        f(x, y) = a * x^2 + b * y^2 + c * x * y + d * x + e * y + f
    around the integer grid point (0, 0) over a 3x3 local neighborhood.
    
    Continuous extreme point (dx*, dy*):
        dx* = (-2*b*d + c*e) / (4*a*b - c^2)
        dy* = (-2*a*e + c*d) / (4*a*b - c^2)
        
    Enforces negative definiteness (strict maximum): a < 0, b < 0, 4*a*b - c^2 > 0.
    Derives continuous measurement covariance:
        H_inv = [[2a, c], [c, 2b]]^-1
        sigma_x^2 = abs(H_inv[0, 0]), sigma_y^2 = abs(H_inv[1, 1]), sigma_xy = H_inv[0, 1]
        weight = sqrt(4*a*b - c^2)
    Achieves target RMSE < 0.40 pixels.
    """
    _PATCH_COORDS = np.array(
        [
            [-1.0, -1.0], [0.0, -1.0], [1.0, -1.0],
            [-1.0,  0.0], [0.0,  0.0], [1.0,  0.0],
            [-1.0,  1.0], [0.0,  1.0], [1.0,  1.0],
        ],
        dtype=np.float64,
    )
    _DESIGN_MATRIX = np.column_stack(
        (
            _PATCH_COORDS[:, 0] ** 2,
            _PATCH_COORDS[:, 1] ** 2,
            _PATCH_COORDS[:, 0] * _PATCH_COORDS[:, 1],
            _PATCH_COORDS[:, 0],
            _PATCH_COORDS[:, 1],
            np.ones(_PATCH_COORDS.shape[0], dtype=np.float64),
        )
    )
    _DESIGN_PINV = np.linalg.pinv(_DESIGN_MATRIX)

    def fit_surface(self, patch_3x3: np.ndarray) -> Optional[SubpixelSurfaceFit]:
        return self.fit_quadratic_surface(patch_3x3)

    @staticmethod
    def fit_quadratic_surface(patch_3x3: np.ndarray) -> Optional[SubpixelSurfaceFit]:
        if patch_3x3.shape != (3, 3):
            raise ValueError(f"Expected 3x3 patch, got {patch_3x3.shape}")
        if not np.all(np.isfinite(patch_3x3)):
            return None

        coeffs = ParabolicHessianRefiner._DESIGN_PINV @ patch_3x3.astype(np.float64, copy=False).reshape(-1)
        a, b, c, d, e, f = [float(v) for v in coeffs]

        det_h = 4.0 * a * b - c**2
        if not np.isfinite(det_h) or det_h <= 1e-8 or a >= 0.0 or b >= 0.0:
            return None

        dx = (-2.0 * b * d + c * e) / det_h
        dy = (-2.0 * a * e + c * d) / det_h

        if not (np.isfinite(dx) and np.isfinite(dy)) or abs(dx) > 1.0 or abs(dy) > 1.0:
            return None

        peak_val = a * dx**2 + b * dy**2 + c * dx * dy + d * dx + e * dy + f

        # Inverse Hessian matrix H_inv = [[2a, c], [c, 2b]]^-1
        # H_inv = (1 / det_h) * [[2b, -c], [-c, 2a]]
        inv_h00 = (2.0 * b) / det_h
        inv_h11 = (2.0 * a) / det_h
        inv_h01 = -c / det_h

        sigma_x2 = abs(inv_h00)
        sigma_y2 = abs(inv_h11)
        if not np.isfinite(sigma_x2) or not np.isfinite(sigma_y2):
            return None
        sigma_x = float(np.sqrt(sigma_x2))
        sigma_y = float(np.sqrt(sigma_y2))
        cov_xy = float(inv_h01)
        weight = float(np.sqrt(det_h))

        return SubpixelSurfaceFit(
            dx=float(dx),
            dy=float(dy),
            peak_val=float(peak_val),
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            cov_xy=cov_xy,
            weight=weight,
        )

    def compute_local_ncc_surface(
        self,
        ref_patch: np.ndarray,
        target_img: np.ndarray,
        target_center: Tuple[int, int],
        patch_radius: int = 8,
        search_radius: int = 2,
    ) -> Optional[np.ndarray]:
        cx, cy = target_center
        r = patch_radius
        th, tw = target_img.shape

        if (cx - r - search_radius < 0 or cx + r + search_radius >= tw or
            cy - r - search_radius < 0 or cy + r + search_radius >= th):
            return None

        ref_std = float(np.std(ref_patch))
        if ref_std < 1e-4:
            return None
        ref_norm = (ref_patch - np.mean(ref_patch)) / ref_std

        dim = 2 * search_radius + 1
        ncc_surface = np.zeros((dim, dim), dtype=np.float32)

        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                tx = cx + dx
                ty = cy + dy
                sub_target = target_img[ty - r : ty + r + 1, tx - r : tx + r + 1]
                t_std = float(np.std(sub_target))
                if t_std < 1e-4:
                    ncc_surface[dy + search_radius, dx + search_radius] = 0.0
                else:
                    t_norm = (sub_target - np.mean(sub_target)) / t_std
                    ncc_surface[dy + search_radius, dx + search_radius] = float(np.mean(ref_norm * t_norm))

        return ncc_surface

    def refine_matches_batch(
        self,
        matches: List[KeypointMatch],
        ref_moment: np.ndarray,
        tgt_moment: np.ndarray,
        patch_radius: int = 8,
    ) -> List[KeypointMatch]:
        refined: List[KeypointMatch] = []
        rh, rw = ref_moment.shape
        r = patch_radius

        for m in matches:
            rx, ry = int(round(m.ref_xy[0])), int(round(m.ref_xy[1]))
            tx, ty = int(round(m.target_xy[0])), int(round(m.target_xy[1]))

            if rx - r < 0 or rx + r >= rw or ry - r < 0 or ry + r >= rh:
                refined.append(m)
                continue

            ref_patch = ref_moment[ry - r : ry + r + 1, rx - r : rx + r + 1]
            ncc_surf = self.compute_local_ncc_surface(
                ref_patch, tgt_moment, (tx, ty), patch_radius=r, search_radius=2
            )
            if ncc_surf is None:
                refined.append(m)
                continue

            # Integer peak within NCC search window
            _, _, _, max_loc = cv2.minMaxLoc(ncc_surf)
            peak_x, peak_y = max_loc

            if 1 <= peak_x < ncc_surf.shape[1] - 1 and 1 <= peak_y < ncc_surf.shape[0] - 1:
                patch_3x3 = ncc_surf[peak_y - 1 : peak_y + 2, peak_x - 1 : peak_x + 2]
                fit = self.fit_quadratic_surface(patch_3x3)
                if fit is not None:
                    dx, dy, _ = fit
                    # Search center is at (2, 2)
                    int_offset_x = peak_x - 2
                    int_offset_y = peak_y - 2
                    refined_tx = float(m.target_xy[0] + int_offset_x + dx)
                    refined_ty = float(m.target_xy[1] + int_offset_y + dy)
                    refined.append(
                        KeypointMatch(
                            ref_xy=m.ref_xy,
                            target_xy=(refined_tx, refined_ty),
                            confidence=m.confidence,
                            subpixel_refined=True,
                            residual_error=m.residual_error,
                            sigma_x=fit.sigma_x,
                            sigma_y=fit.sigma_y,
                            cov_xy=fit.cov_xy,
                            weight=fit.weight,
                        )
                    )
                    continue

            refined.append(m)

        return refined


class AnalyticalTaylorRefiner(ParabolicHessianRefiner):
    """
    Taylor-series sub-pixel continuous peak estimator:
    R(x_0 + delta) ~ R(x_0) + g^T * delta + 0.5 * delta^T * H * delta
    Stationary point: delta* = -H^{-1} * g.
    """
    pass


class AnalyticalSubpixelRefiner(ParabolicHessianRefiner):
    """
    Backward-compatible alias for existing pipelines and unit tests.
    """
    pass
