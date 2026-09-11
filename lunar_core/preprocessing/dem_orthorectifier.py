"""
DEM-Based Orthorectification and 3D Topographic Terrain Parallax Engine.
SIH PS 26166: Multi-modal, Sun angle and scale invariant image correspondence.

Resolves severe 3D lunar relief displacement and crater rim parallax that break
single global 2D planar homography assumptions in deep south-polar crater terrain.
Explicitly distinguishes DEM_ORTHORECTIFIED geometry from PLANAR_APPROXIMATION.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np


@dataclass
class LunarDEM:
    """
    Digital Elevation Model abstraction representing lunar topography.
    """
    elevation_m: np.ndarray
    pixel_size_m: float = 5.0
    origin_x_m: float = 0.0
    origin_y_m: float = 0.0
    nodata_value: float = -9999.0
    name: str = "lunar_topography_dem"

    @property
    def shape(self) -> Tuple[int, int]:
        return self.elevation_m.shape[:2]

    def get_elevation_bilinear(self, x_m: float, y_m: float) -> float:
        """Sample continuous elevation at map coordinates (x_m, y_m)."""
        col = (x_m - self.origin_x_m) / self.pixel_size_m
        row = (y_m - self.origin_y_m) / self.pixel_size_m
        h, w = self.shape

        if col < 0 or col >= w - 1 or row < 0 or row >= h - 1:
            return float(np.nanmedian(self.elevation_m))

        c0, r0 = int(np.floor(col)), int(np.floor(row))
        c1, r1 = c0 + 1, r0 + 1
        dc, dr = col - c0, row - r0

        v00 = float(self.elevation_m[r0, c0])
        v01 = float(self.elevation_m[r0, c1])
        v10 = float(self.elevation_m[r1, c0])
        v11 = float(self.elevation_m[r1, c1])

        # Ignore nodata
        vals = [v for v in (v00, v01, v10, v11) if v != self.nodata_value and np.isfinite(v)]
        if len(vals) < 4:
            return float(np.median(vals)) if vals else 0.0

        top = (1.0 - dc) * v00 + dc * v01
        bot = (1.0 - dc) * v10 + dc * v11
        return float((1.0 - dr) * top + dr * bot)


def _default_nadir_rotation() -> np.ndarray:
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float64)


@dataclass
class CameraGeometryModel:
    """
    Planetary pushbroom / framing camera pinhole model with extrinsic orbit state.
    """
    focal_length_px: float
    principal_point_px: Tuple[float, float]
    spacecraft_pos_m: Tuple[float, float, float]  # (X_c, Y_c, Z_c) altitude
    rotation_matrix: np.ndarray = field(default_factory=_default_nadir_rotation)
    sensor_name: str = "OHRC_FRAMING_EQUIVALENT"

    def project_surface_to_image(self, x_m: float, y_m: float, z_m: float) -> Tuple[float, float, bool]:
        """Projects 3D lunar surface coordinate (X, Y, Z) in meters into sensor pixel (u, v)."""
        # Vector from camera to ground point
        p_ground = np.array([x_m, y_m, z_m], dtype=np.float64)
        c_pos = np.array(self.spacecraft_pos_m, dtype=np.float64)
        v_world = p_ground - c_pos

        # Transform into camera frame
        v_cam = self.rotation_matrix.T @ v_world

        # Points behind camera plane are invalid
        if v_cam[2] <= 1.0:
            return 0.0, 0.0, False

        cx, cy = self.principal_point_px
        u = cx + (self.focal_length_px * v_cam[0]) / v_cam[2]
        v = cy + (self.focal_length_px * v_cam[1]) / v_cam[2]
        return float(u), float(v), True


class DEMOrthorectifier:
    """
    Orthorectifies raw planetary rasters into map-projected surface grids using DEMs,
    suppressing parallax relief displacement before feature matching.
    """

    @staticmethod
    def orthorectify_image(
        image: np.ndarray,
        camera: CameraGeometryModel,
        dem: Optional[LunarDEM],
        output_grid_shape: Optional[Tuple[int, int]] = None,
        output_gsd_m: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Orthorectifies image onto map terrain grid. If dem is None, returns planar approximation.
        """
        img_arr = np.asarray(image, dtype=np.float32)
        h_img, w_img = img_arr.shape[:2]

        if dem is None:
            # Planar approximation fallback
            return img_arr.copy(), {
                "terrain_model": "PLANAR_APPROXIMATION",
                "dem_used": False,
                "dem_name": None,
                "mean_elevation_m": 0.0,
                "relief_amplitude_m": 0.0,
                "max_parallax_displacement_px": 0.0,
                "status": "PLANAR_APPROXIMATION_FALLBACK",
            }

        h_dem, w_dem = dem.shape
        out_h = output_grid_shape[0] if output_grid_shape else h_dem
        out_w = output_grid_shape[1] if output_grid_shape else w_dem
        pixel_size = output_gsd_m if output_gsd_m else dem.pixel_size_m

        ortho_img = np.zeros((out_h, out_w), dtype=np.float32)
        parallax_displacements = []

        # Reference datum elevation (median surface)
        datum_z = float(np.nanmedian(dem.elevation_m))
        z_min, z_max = float(np.nanmin(dem.elevation_m)), float(np.nanmax(dem.elevation_m))

        # Vectorized coordinate grids
        cols, rows = np.meshgrid(np.arange(out_w), np.arange(out_h))
        x_map = dem.origin_x_m + cols * pixel_size
        y_map = dem.origin_y_m + rows * pixel_size

        # Subsample map grid for fast coordinate warping
        step = max(1, min(out_h, out_w) // 128)
        sub_rows = np.arange(0, out_h, step)
        sub_cols = np.arange(0, out_w, step)
        if sub_rows[-1] != out_h - 1:
            sub_rows = np.append(sub_rows, out_h - 1)
        if sub_cols[-1] != out_w - 1:
            sub_cols = np.append(sub_cols, out_w - 1)

        sub_c_grid, sub_r_grid = np.meshgrid(sub_cols, sub_rows)
        sub_x = dem.origin_x_m + sub_c_grid * pixel_size
        sub_y = dem.origin_y_m + sub_r_grid * pixel_size

        map_u = np.zeros(sub_c_grid.shape, dtype=np.float32)
        map_v = np.zeros(sub_r_grid.shape, dtype=np.float32)

        for r_idx in range(len(sub_rows)):
            for c_idx in range(len(sub_cols)):
                xm = sub_x[r_idx, c_idx]
                ym = sub_y[r_idx, c_idx]
                zm = dem.get_elevation_bilinear(xm, ym)
                u, v, valid = camera.project_surface_to_image(xm, ym, zm)

                # Planar reference comparison for parallax calculation
                u_plan, v_plan, _ = camera.project_surface_to_image(xm, ym, datum_z)
                if valid:
                    parallax_displacements.append(np.sqrt((u - u_plan)**2 + (v - v_plan)**2))

                map_u[r_idx, c_idx] = u if valid else -1.0
                map_v[r_idx, c_idx] = v if valid else -1.0

        # Upsample dense mapping
        dense_map_x = cv2.resize(map_u, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        dense_map_y = cv2.resize(map_v, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        # Remap image pixels onto map terrain grid
        ortho_img = cv2.remap(
            img_arr,
            dense_map_x,
            dense_map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        max_disp = float(np.max(parallax_displacements)) if parallax_displacements else 0.0
        mean_disp = float(np.mean(parallax_displacements)) if parallax_displacements else 0.0

        provenance = {
            "terrain_model": "DEM_ORTHORECTIFIED",
            "dem_used": True,
            "dem_name": dem.name,
            "mean_elevation_m": datum_z,
            "relief_amplitude_m": z_max - z_min,
            "max_parallax_displacement_px": max_disp,
            "mean_parallax_displacement_px": mean_disp,
            "status": "DEM_ORTHORECTIFIED_SUCCESS",
        }
        return ortho_img, provenance
