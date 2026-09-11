"""
Tests for DEM-Based Orthorectification and 3D Topographic Parallax Engine (SIH PS 26166).
"""

from __future__ import annotations

import numpy as np
import pytest

from lunar_core.preprocessing.dem_orthorectifier import (
    CameraGeometryModel,
    DEMOrthorectifier,
    LunarDEM,
)


def test_dem_orthorectifier_planar_fallback():
    img = np.ones((64, 64), dtype=np.float32) * 120.0
    cam = CameraGeometryModel(
        focal_length_px=1000.0,
        principal_point_px=(32.0, 32.0),
        spacecraft_pos_m=(0.0, 0.0, 100000.0),
    )

    ortho, meta = DEMOrthorectifier.orthorectify_image(img, cam, dem=None)
    assert ortho.shape == (64, 64)
    assert np.allclose(ortho, img)
    assert meta["terrain_model"] == "PLANAR_APPROXIMATION"
    assert meta["dem_used"] is False


def test_dem_orthorectifier_with_topography():
    # Synthetic 64x64 DEM with a central crater depression of 500m
    elev = np.zeros((64, 64), dtype=np.float32)
    y, x = np.ogrid[:64, :64]
    dist_sq = (x - 32) ** 2 + (y - 32) ** 2
    elev[dist_sq < 200] = -500.0  # 500m deep crater

    dem = LunarDEM(
        elevation_m=elev,
        pixel_size_m=10.0,
        origin_x_m=-320.0,
        origin_y_m=-320.0,
        name="synthetic_crater_dem",
    )

    # Test continuous elevation interpolation
    z_center = dem.get_elevation_bilinear(0.0, 0.0)
    assert z_center < -400.0  # inside crater depression

    img = np.random.RandomState(42).uniform(50.0, 200.0, (64, 64)).astype(np.float32)
    cam = CameraGeometryModel(
        focal_length_px=2000.0,
        principal_point_px=(32.0, 32.0),
        spacecraft_pos_m=(50.0, 0.0, 50000.0),  # Off-nadir camera creates parallax
    )

    ortho, meta = DEMOrthorectifier.orthorectify_image(
        image=img,
        camera=cam,
        dem=dem,
        output_grid_shape=(64, 64),
        output_gsd_m=10.0,
    )

    assert ortho.shape == (64, 64)
    assert meta["terrain_model"] == "DEM_ORTHORECTIFIED"
    assert meta["dem_used"] is True
    assert meta["relief_amplitude_m"] == 500.0
    assert meta["max_parallax_displacement_px"] > 0.0
