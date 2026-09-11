"""
Tests for Constrained Local Tiled Registration and Residual Parallax Estimation (SIH PS 26166).
"""

from __future__ import annotations

import numpy as np
import pytest

from lunar_core.models import KeypointMatch
from samanvaya.registration.local_tiles import (
    LocalTiledRegistrationEngine,
    LocalTiledRegistrationResult,
)


def test_local_tiles_insufficient_inliers_fallback():
    engine = LocalTiledRegistrationEngine(grid_rows=2, grid_cols=2, min_inliers_per_tile=4)
    # Only 2 matches total across a 100x100 image
    matches = [
        KeypointMatch(target_xy=(10.0, 10.0), ref_xy=(12.0, 11.0), confidence=0.9),
        KeypointMatch(target_xy=(20.0, 20.0), ref_xy=(22.0, 21.0), confidence=0.9),
    ]
    H_global = np.eye(3)
    H_global[0, 2] = 2.0
    H_global[1, 2] = 1.0

    res = engine.estimate_local_models(matches, (100, 100), H_global)
    assert isinstance(res, LocalTiledRegistrationResult)
    assert res.grid_rows == 2
    assert res.grid_cols == 2
    assert res.fallback_tiles_count == 4
    assert res.valid_tiles_count == 0
    assert res.regularization_status == "ALL_TILES_GLOBAL_FALLBACK"
    assert res.displacement_field_dx.shape == (2, 2)
    assert res.displacement_field_dy.shape == (2, 2)


def test_local_tiles_estimation_with_distributed_inliers():
    engine = LocalTiledRegistrationEngine(
        grid_rows=2,
        grid_cols=2,
        min_inliers_per_tile=4,
        max_condition_number=50.0,
        max_deformation_threshold_px=10.0,
    )

    rng = np.random.RandomState(42)
    matches = []
    # Populate all 4 quadrants with 6 matches each
    # Quadrant ranges:
    # Q0: x in [5, 45], y in [5, 45]
    # Q1: x in [55, 95], y in [5, 45]
    # Q2: x in [5, 45], y in [55, 95]
    # Q3: x in [55, 95], y in [55, 95]
    ranges = [
        ((5, 45), (5, 45)),
        ((55, 95), (5, 45)),
        ((5, 45), (55, 95)),
        ((55, 95), (55, 95)),
    ]

    for (xr, yr) in ranges:
        for _ in range(8):
            sx = rng.uniform(xr[0], xr[1])
            sy = rng.uniform(yr[0], yr[1])
            # slight translation + small noise
            rx = sx + 3.0 + rng.normal(0, 0.1)
            ry = sy - 2.0 + rng.normal(0, 0.1)
            matches.append(KeypointMatch(target_xy=(sx, sy), ref_xy=(rx, ry), confidence=0.9))

    H_global = np.eye(3)
    H_global[0, 2] = 3.0
    H_global[1, 2] = -2.0

    res = engine.estimate_local_models(matches, (100, 100), H_global)
    assert res.valid_tiles_count > 0
    assert res.mean_local_rmse < 0.50
    assert res.regularization_status == "CONSTRAINED_LOCAL_TILES_ESTIMATED"
    d = res.to_dict()
    assert "mean_local_rmse_px" in d
    assert "tile_rmse_grid" in d
