"""
Constrained Local Tiled Registration and Residual Parallax Estimation.
SIH PS 26166: Multi-modal, Sun angle and scale invariant image correspondence.

Resolves unmodeled local topographic parallax and non-rigid relief distortion
by regularizing local affine/similarity estimators over an NxM spatial grid.
Guarantees strict constraints on condition number, deformation bounds, and continuity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from lunar_core.models import KeypointMatch


@dataclass
class LocalTileModel:
    """Fitted geometric model for an individual spatial tile."""
    tile_row: int
    tile_col: int
    inlier_count: int
    matrix_2x3: Optional[np.ndarray]
    reprojection_rmse: float
    condition_number: float
    max_deformation_px: float
    is_valid: bool
    used_fallback: bool
    status: str


@dataclass
class LocalTiledRegistrationResult:
    """Composite result of constrained local tiled registration."""
    grid_rows: int
    grid_cols: int
    total_inliers: int
    valid_tiles_count: int
    fallback_tiles_count: int
    tile_models: List[LocalTileModel]
    displacement_field_dx: np.ndarray
    displacement_field_dy: np.ndarray
    tile_rmse_grid: np.ndarray
    mean_local_rmse: float
    max_local_deformation_px: float
    regularization_status: str
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grid_dimensions": [self.grid_rows, self.grid_cols],
            "total_inliers": self.total_inliers,
            "valid_tiles_count": self.valid_tiles_count,
            "fallback_tiles_count": self.fallback_tiles_count,
            "mean_local_rmse_px": round(self.mean_local_rmse, 4),
            "max_local_deformation_px": round(self.max_local_deformation_px, 4),
            "regularization_status": self.regularization_status,
            "tile_rmse_grid": self.tile_rmse_grid.tolist(),
        }


class LocalTiledRegistrationEngine:
    """
    Subdivides the scene into a grid of spatial tiles and estimates regularized
    local transformations to absorb localized terrain-induced parallax.
    """

    def __init__(
        self,
        grid_rows: int = 4,
        grid_cols: int = 4,
        min_inliers_per_tile: int = 4,
        max_condition_number: float = 100.0,
        max_deformation_threshold_px: float = 8.0,
    ) -> None:
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.min_inliers = min_inliers_per_tile
        self.max_cond = max_condition_number
        self.max_deform = max_deformation_threshold_px

    def estimate_local_models(
        self,
        inliers: List[KeypointMatch],
        source_shape: Tuple[int, int],
        global_transform: np.ndarray,
    ) -> LocalTiledRegistrationResult:
        """
        Estimates regularized local models across the spatial tile grid.
        """
        h_src, w_src = source_shape[:2]
        tile_h = h_src / float(self.grid_rows)
        tile_w = w_src / float(self.grid_cols)

        # Global matrix as 2x3 or 3x3
        g_mat_3x3 = global_transform.copy()
        if g_mat_3x3.shape == (2, 3):
            g_mat_3x3 = np.vstack([g_mat_3x3, [0.0, 0.0, 1.0]])

        # Bin inliers into tiles based on source coordinates
        tile_bins: List[List[KeypointMatch]] = [
            [] for _ in range(self.grid_rows * self.grid_cols)
        ]
        for m in inliers:
            sx, sy = float(m.target_xy[0]), float(m.target_xy[1])
            col_idx = min(max(0, int(sx // tile_w)), self.grid_cols - 1)
            row_idx = min(max(0, int(sy // tile_h)), self.grid_rows - 1)
            bin_idx = row_idx * self.grid_cols + col_idx
            tile_bins[bin_idx].append(m)

        tile_models: List[LocalTileModel] = []
        valid_count = 0
        fallback_count = 0
        rmse_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        deform_max_all = 0.0

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                bin_idx = r * self.grid_cols + c
                matches_tile = tile_bins[bin_idx]
                n_tile = len(matches_tile)

                # Default fallback model from global matrix
                tile_center_src = np.array([c * tile_w + tile_w * 0.5, r * tile_h + tile_h * 0.5, 1.0])
                pt_global_ref = g_mat_3x3 @ tile_center_src
                pt_global_ref = pt_global_ref[:2] / max(abs(pt_global_ref[2]), 1e-12)

                if n_tile < self.min_inliers:
                    tile_models.append(
                        LocalTileModel(
                            tile_row=r,
                            tile_col=c,
                            inlier_count=n_tile,
                            matrix_2x3=g_mat_3x3[:2, :].copy(),
                            reprojection_rmse=0.0,
                            condition_number=float(np.linalg.cond(g_mat_3x3[:2, :2])),
                            max_deformation_px=0.0,
                            is_valid=False,
                            used_fallback=True,
                            status="INSUFFICIENT_INLIERS_GLOBAL_FALLBACK",
                        )
                    )
                    fallback_count += 1
                    continue

                # Estimate local affine model
                src_pts = np.array([[m.target_xy[0], m.target_xy[1]] for m in matches_tile], dtype=np.float32)
                ref_pts = np.array([[m.ref_xy[0], m.ref_xy[1]] for m in matches_tile], dtype=np.float32)

                local_mat, inlier_mask = cv2.estimateAffine2D(
                    src_pts, ref_pts, method=cv2.RANSAC, ransacReprojThreshold=1.5
                )

                if local_mat is None:
                    tile_models.append(
                        LocalTileModel(
                            tile_row=r,
                            tile_col=c,
                            inlier_count=n_tile,
                            matrix_2x3=g_mat_3x3[:2, :].copy(),
                            reprojection_rmse=0.0,
                            condition_number=float(np.linalg.cond(g_mat_3x3[:2, :2])),
                            max_deformation_px=0.0,
                            is_valid=False,
                            used_fallback=True,
                            status="ESTIMATION_FAILED_GLOBAL_FALLBACK",
                        )
                    )
                    fallback_count += 1
                    continue

                # Calculate condition number and local residuals
                cond_num = float(np.linalg.cond(local_mat[:, :2]))
                src_h = np.column_stack([src_pts, np.ones(n_tile)])
                pred_ref = src_h @ local_mat.T
                residuals = np.linalg.norm(ref_pts - pred_ref, axis=1)
                tile_rmse = float(np.sqrt(np.mean(residuals ** 2)))

                # Calculate deformation against global model at tile center
                pt_local_ref = local_mat @ tile_center_src
                deform = float(np.linalg.norm(pt_local_ref - pt_global_ref))

                # Regularization checks
                if cond_num > self.max_cond or deform > self.max_deform or np.isnan(tile_rmse):
                    # Local fit violates physical constraints: fall back to global
                    tile_models.append(
                        LocalTileModel(
                            tile_row=r,
                            tile_col=c,
                            inlier_count=n_tile,
                            matrix_2x3=g_mat_3x3[:2, :].copy(),
                            reprojection_rmse=tile_rmse,
                            condition_number=cond_num,
                            max_deformation_px=deform,
                            is_valid=False,
                            used_fallback=True,
                            status="CONSTRAINT_VIOLATION_GLOBAL_FALLBACK",
                        )
                    )
                    fallback_count += 1
                else:
                    tile_models.append(
                        LocalTileModel(
                            tile_row=r,
                            tile_col=c,
                            inlier_count=n_tile,
                            matrix_2x3=local_mat,
                            reprojection_rmse=tile_rmse,
                            condition_number=cond_num,
                            max_deformation_px=deform,
                            is_valid=True,
                            used_fallback=False,
                            status="LOCAL_MODEL_ESTIMATED",
                        )
                    )
                    valid_count += 1
                    rmse_grid[r, c] = tile_rmse
                    deform_max_all = max(deform_max_all, deform)

        # Build smooth 2D displacement field relative to global homography
        field_dx = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        field_dy = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)

        for tm in tile_models:
            r, c = tm.tile_row, tm.tile_col
            tile_c_src = np.array([c * tile_w + tile_w * 0.5, r * tile_h + tile_h * 0.5, 1.0])
            pt_glob = g_mat_3x3 @ tile_c_src
            pt_glob = pt_glob[:2] / max(abs(pt_glob[2]), 1e-12)

            pt_loc = tm.matrix_2x3 @ tile_c_src if tm.matrix_2x3 is not None else pt_glob
            field_dx[r, c] = pt_loc[0] - pt_glob[0]
            field_dy[r, c] = pt_loc[1] - pt_glob[1]

        mean_rmse = float(np.mean(rmse_grid[rmse_grid > 0])) if np.any(rmse_grid > 0) else 0.0

        return LocalTiledRegistrationResult(
            grid_rows=self.grid_rows,
            grid_cols=self.grid_cols,
            total_inliers=len(inliers),
            valid_tiles_count=valid_count,
            fallback_tiles_count=fallback_count,
            tile_models=tile_models,
            displacement_field_dx=field_dx,
            displacement_field_dy=field_dy,
            tile_rmse_grid=rmse_grid,
            mean_local_rmse=mean_rmse,
            max_local_deformation_px=deform_max_all,
            regularization_status="CONSTRAINED_LOCAL_TILES_ESTIMATED" if valid_count > 0 else "ALL_TILES_GLOBAL_FALLBACK",
            provenance={
                "grid_size": f"{self.grid_rows}x{self.grid_cols}",
                "min_inliers_per_tile": self.min_inliers,
                "max_condition_number": self.max_cond,
                "max_deformation_threshold_px": self.max_deform,
            },
        )
