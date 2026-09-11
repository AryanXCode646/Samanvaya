"""
Tests for Pre-Registration Quality Gates (SIH PS 26166).
"""

from __future__ import annotations

import numpy as np
import pytest

from samanvaya.validation.quality_gates import (
    QualityGateResult,
    compute_image_quality_metrics,
    evaluate_pre_registration_gates,
)


def test_quality_gates_null_and_empty():
    res_none = evaluate_pre_registration_gates(None, np.ones((32, 32)))
    assert res_none.passed is False
    assert res_none.rejection_code == "DATA_REQUIRED"

    res_empty = evaluate_pre_registration_gates(np.array([]), np.ones((32, 32)))
    assert res_empty.passed is False
    assert res_empty.rejection_code == "DATA_REQUIRED"


def test_quality_gates_invalid_geometry():
    tiny = np.ones((8, 8), dtype=np.float32)
    valid = np.ones((32, 32), dtype=np.float32)
    res = evaluate_pre_registration_gates(tiny, valid, min_dim=16)
    assert res.passed is False
    assert res.rejection_code == "INVALID_GEOMETRY"


def test_quality_gates_overlap_threshold():
    img1 = np.random.RandomState(1).uniform(0, 100, (32, 32)).astype(np.float32)
    img2 = np.random.RandomState(2).uniform(0, 100, (32, 32)).astype(np.float32)

    # 0% overlap
    res_zero = evaluate_pre_registration_gates(img1, img2, overlap_ratio=0.0)
    assert res_zero.passed is False
    assert res_zero.rejection_code == "NON_OVERLAPPING"

    # Below 5% threshold
    res_low = evaluate_pre_registration_gates(img1, img2, overlap_ratio=0.02)
    assert res_low.passed is False
    assert res_low.rejection_code == "NON_OVERLAPPING"


def test_quality_gates_texture_and_shadow():
    flat_src = np.ones((64, 64), dtype=np.float32) * 50.0  # Zero variance
    ref = np.random.RandomState(1).uniform(10, 200, (64, 64)).astype(np.float32)

    res_flat = evaluate_pre_registration_gates(flat_src, ref)
    assert res_flat.passed is False
    assert res_flat.rejection_code == "LOW_TEXTURE"

    # High NaN fraction
    nan_src = np.random.RandomState(2).uniform(10, 200, (64, 64)).astype(np.float32)
    nan_src[:60, :] = np.nan  # > 90% NaN
    res_nan = evaluate_pre_registration_gates(nan_src, ref, min_valid_fraction=0.20)
    assert res_nan.passed is False
    assert res_nan.rejection_code == "INSUFFICIENT_VALID_AREA"


def test_quality_gates_success():
    src = np.random.RandomState(10).uniform(20, 220, (64, 64)).astype(np.float32)
    ref = np.random.RandomState(20).uniform(20, 220, (64, 64)).astype(np.float32)

    res = evaluate_pre_registration_gates(src, ref, overlap_ratio=0.65)
    assert res.passed is True
    assert res.rejection_code is None
    assert "src_variance" in res.metrics
    assert res.metrics["overlap_ratio"] == 0.65
