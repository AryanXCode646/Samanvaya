"""
Tests for Spectral Cascade Composition and Uncertainty Propagation (SIH PS 26166).
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from lunar_core.alignment.spectral_cascade import (
    CascadeResult,
    CascadeStage,
    IIRS_REGISTRATION_STATUS,
    SpectralCascadeEngine,
)


def test_spectral_cascade_composition():
    engine = SpectralCascadeEngine()

    # Stage 1: IIRS -> TMC-2 (scale factor = 16.0, translation = (10, 20), rmse = 0.5)
    H1 = np.array([
        [16.0, 0.0, 10.0],
        [0.0, 16.0, 20.0],
        [0.0, 0.0, 1.0],
    ])
    s1 = CascadeStage(
        source_instrument="IIRS",
        reference_instrument="TMC-2",
        transform_matrix=H1,
        residual_rmse=0.50,
        scale_factor=16.0,
    )

    # Stage 2: TMC-2 -> OHRC (scale factor = 20.0, translation = (5, 5), rmse = 0.3)
    H2 = np.array([
        [20.0, 0.0, 5.0],
        [0.0, 20.0, 5.0],
        [0.0, 0.0, 1.0],
    ])
    s2 = CascadeStage(
        source_instrument="TMC-2",
        reference_instrument="OHRC",
        transform_matrix=H2,
        residual_rmse=0.30,
        scale_factor=20.0,
    )

    result = engine.compose([s1, s2])
    assert isinstance(result, CascadeResult)
    assert result.status == IIRS_REGISTRATION_STATUS
    assert len(result.stages) == 2

    # Verify composed transform: H_comp = H2 @ H1
    # [20, 0, 5] @ [16, 0, 10; 0, 16, 20; 0, 0, 1]
    # x_ohrc = 20 * (16 * x + 10) + 5 = 320 * x + 205
    expected_H = H2 @ H1
    assert np.allclose(result.composed_transform, expected_H)

    # Verify uncertainty propagation:
    # Downstream scale of Stage 2 is 20.0
    # sigma_eff = sqrt((20.0 * 0.50)^2 + 0.30^2) = sqrt(100.0 + 0.09) = sqrt(100.09) approx 10.0045
    expected_sigma = math.sqrt((20.0 * 0.50) ** 2 + 0.30 ** 2)
    assert math.isclose(result.effective_uncertainty_px, expected_sigma, rel_tol=1e-4)

    # Verify composed covariance
    assert result.composed_covariance is not None
    assert result.composed_covariance.shape == (2, 2)
    expected_cov = (20.0 ** 2) * (0.50 ** 2) * np.eye(2) + (0.30 ** 2) * np.eye(2)
    assert np.allclose(result.composed_covariance, expected_cov)


def test_spectral_cascade_invalid_stage_handling():
    engine = SpectralCascadeEngine()
    s1 = CascadeStage(
        source_instrument="IIRS",
        reference_instrument="TMC-2",
        transform_matrix=np.eye(3),
        residual_rmse=0.5,
        valid=False,  # marked invalid
    )
    result = engine.compose([s1])
    assert result.status == "FAILED_STAGE_INVALID"
    assert math.isinf(result.effective_uncertainty_px)
