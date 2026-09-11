"""
Regression test for SIH PS 26166 Phase 11: Match Distribution on SOURCE IMAGE ONLY.
Verifies that spatial selection operates in FULL_SOURCE_IMAGE coordinates,
and correctly rebalances/caps matches when reference distribution is uniform but
source distribution is heavily clustered.
"""

from __future__ import annotations

import numpy as np
import pytest

from lunar_core.models import KeypointMatch
from lunar_core.postprocessing.anms import SpatialUniformDistributor


def test_source_space_rebalances_clustered_source_points():
    """
    Test scenario:
    - 64 matches with reference points evenly distributed across all 64 grid cells (1 per cell).
    - But ALL 64 matches have source coordinates clustered into cell (0, 0) [e.g. coordinates (2.0, 2.0)].
    Under source-based spatial selection, capping per cell (e.g. cap=4) must rebalance
    and reject 60 of the clustered points, leaving at most 4 points in that source cell!
    """
    distributor = SpatialUniformDistributor(grid_rows=8, grid_cols=8)
    source_shape = (64, 64)
    ref_shape = (64, 64)

    matches = []
    for gy in range(8):
        for gx in range(8):
            ref_x = float(gx * 8 + 4)
            ref_y = float(gy * 8 + 4)
            # All source points clumped at (2.0, 2.0) in the top-left cell
            src_x = 2.0 + np.random.uniform(-0.5, 0.5)
            src_y = 2.0 + np.random.uniform(-0.5, 0.5)
            matches.append(
                KeypointMatch(
                    ref_xy=(ref_x, ref_y),
                    target_xy=(src_x, src_y),
                    confidence=0.95 - (gy * 8 + gx) * 0.005,
                )
            )

    assert len(matches) == 64

    # Cap grid cells in SOURCE frame
    capped = distributor.cap_grid_cells(matches, source_shape, cap_per_cell=4, use_source_coords=True)
    assert len(capped) == 4, f"Expected 4 capped matches from source cell (0, 0), got {len(capped)}"

    # Metrics on the raw clustered source points
    raw_metrics = distributor.compute_spatial_metrics(matches, source_shape, use_source_coords=True)
    assert raw_metrics["occupied_cells"] == 1
    assert raw_metrics["max_cluster_fraction"] == 1.0
    assert raw_metrics["spatial_entropy"] == 0.0
    assert raw_metrics["coverage_fraction"] == 1.0 / 64.0

    # Reference frame metrics would falsely show uniform distribution
    ref_metrics = distributor.compute_spatial_metrics(matches, ref_shape, use_source_coords=False)
    assert ref_metrics["occupied_cells"] == 64
    assert ref_metrics["max_cluster_fraction"] == 1.0 / 64.0
    assert ref_metrics["spatial_entropy"] > 0.99
