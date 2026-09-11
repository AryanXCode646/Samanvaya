"""
Tests for Independent Checkpoint Partitioning and Held-Out Validation (SIH PS 26166).
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from lunar_core.models import KeypointMatch
from samanvaya.validation.checkpoints import (
    IndependentValidationReport,
    evaluate_independent_checkpoints,
    partition_control_and_checkpoints,
)


def _make_mock_matches(n: int = 20, noise_std: float = 0.1, seed: int = 42) -> tuple[list[KeypointMatch], np.ndarray]:
    rng = np.random.RandomState(seed)
    # True transformation: affine with scale=1.02, rotation=2 deg, translation=(12.0, -8.0)
    theta = np.deg2rad(2.0)
    scale = 1.02
    A = scale * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    t = np.array([12.0, -8.0])
    H_true = np.eye(3)
    H_true[:2, :2] = A
    H_true[:2, 2] = t

    matches = []
    for i in range(n):
        sx = rng.uniform(10.0, 500.0)
        sy = rng.uniform(10.0, 500.0)
        src_pt = np.array([sx, sy])
        ref_pt = A @ src_pt + t + rng.normal(0.0, noise_std, size=2)
        matches.append(
            KeypointMatch(
                target_xy=(float(sx), float(sy)),
                ref_xy=(float(ref_pt[0]), float(ref_pt[1])),
                confidence=0.95,
            )
        )
    return matches, H_true


def test_partition_control_and_checkpoints_separation():
    matches, _ = _make_mock_matches(n=20)
    control, checkpoints = partition_control_and_checkpoints(matches, control_fraction=0.70, seed=42)

    assert len(control) == 14
    assert len(checkpoints) == 6
    assert len(control) + len(checkpoints) == 20

    # Ensure zero leakage / overlap between control and checkpoints
    control_set = set(id(m) for m in control)
    chk_set = set(id(m) for m in checkpoints)
    assert control_set.isdisjoint(chk_set)


def test_independent_checkpoints_evaluation_metrics():
    matches, H_true = _make_mock_matches(n=30, noise_std=0.15)
    control, checkpoints = partition_control_and_checkpoints(matches, control_fraction=0.70, seed=42)

    report = evaluate_independent_checkpoints(
        transform_matrix=H_true,
        control_points=control,
        checkpoints=checkpoints,
        inlier_ratio=1.0,
        model_type="AFFINE",
    )

    assert isinstance(report, IndependentValidationReport)
    assert report.control_points_count == 21
    assert report.independent_checkpoints_count == 9
    assert report.status == "INDEPENDENTLY_VALIDATED"
    assert 0.05 < report.checkpoint_rmse_pixels < 0.40
    assert report.checkpoint_mae_pixels > 0.0
    assert report.checkpoint_p95_pixels >= report.checkpoint_median_pixels
    assert report.checkpoint_max_pixels >= report.checkpoint_p95_pixels
    assert report.meets_isro_mandate is True


def test_independent_validation_report_export(tmp_path: Path):
    matches, H_true = _make_mock_matches(n=20, noise_std=0.20)
    control, checkpoints = partition_control_and_checkpoints(matches, control_fraction=0.70, seed=42)

    report = evaluate_independent_checkpoints(H_true, control, checkpoints)

    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "report.csv"

    report.export_json(json_path)
    report.export_csv(csv_path)

    assert json_path.is_file()
    assert csv_path.is_file()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["metadata"]["validation_paradigm"] == "STRICT_INDEPENDENT_CHECKPOINTS"
    assert "checkpoint_rmse_pixels" in data["summary"]

    csv_text = csv_path.read_text(encoding="utf-8")
    assert "checkpoint_rmse_pixels" in csv_text
    assert "meets_isro_mandate" in csv_text
