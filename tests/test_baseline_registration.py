"""Tests for classical registration baseline comparison module."""

import numpy as np
import pytest

from samanvaya.validation.baseline_registration import run_baseline_registration


def test_baseline_registration_returns_structured_metrics():
    # Synthetic textured square
    rng = np.random.default_rng(42)
    source = rng.integers(50, 200, size=(128, 128)).astype(np.float32)
    # Add clear features (crosses/dots)
    source[30:35, 30:35] = 255
    source[80:85, 80:85] = 10
    source[20:25, 80:85] = 240
    source[80:85, 20:25] = 20

    # Reference is identical (identity transform)
    reference = source.copy()

    checkpoints = [
        {"point_id": "cp_1", "source_x": 32.0, "source_y": 32.0, "reference_x": 32.0, "reference_y": 32.0},
        {"point_id": "cp_2", "source_x": 82.0, "source_y": 82.0, "reference_x": 82.0, "reference_y": 82.0},
        {"point_id": "cp_3", "source_x": 82.0, "source_y": 22.0, "reference_x": 82.0, "reference_y": 22.0},
        {"point_id": "cp_4", "source_x": 22.0, "source_y": 82.0, "reference_x": 22.0, "reference_y": 82.0},
    ]

    result = run_baseline_registration(source, reference, checkpoints=checkpoints)

    assert "method" in result
    assert "descriptor" in result
    assert "status" in result
    assert result["status"] in {"SUCCESS", "LOW_INLIER_RATIO", "NO_CORRESPONDENCE", "NO_MATCHES"}
    if result["status"] == "SUCCESS":
        assert result["raw_match_count"] >= 4
        assert result["inlier_count"] >= 4
        assert result["rmse_pixels"] is not None
        assert result["rmse_pixels"] >= 0.0
        assert result["coverage_fraction"] >= 0.0


def test_baseline_registration_handles_blank_images():
    source = np.zeros((64, 64), dtype=np.float32)
    reference = np.zeros((64, 64), dtype=np.float32)

    result = run_baseline_registration(source, reference)

    assert result["status"] in {"NO_MATCHES", "NO_CORRESPONDENCE"}
    assert result["inlier_count"] == 0
    assert result["transform_matrix"] is None
