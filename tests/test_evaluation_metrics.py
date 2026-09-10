"""
Comprehensive Test Suite for EvaluationEngine: SIH PS 26166 Metrics and Exports.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from lunar_core.models import KeypointMatch
from lunar_core.evaluation.metrics import EvaluationEngine, RegistrationEvaluationReport


def test_inlier_ratio_computation():
    """Verifies Inlier Ratio (%) = (Inlier Count / Total Matches) * 100."""
    ratio = EvaluationEngine.compute_inlier_ratio(inlier_count=45, total_matches=100)
    assert ratio == 45.0

    ratio_zero = EvaluationEngine.compute_inlier_ratio(inlier_count=0, total_matches=50)
    assert ratio_zero == 0.0

    ratio_empty = EvaluationEngine.compute_inlier_ratio(inlier_count=0, total_matches=0)
    assert ratio_empty == 0.0

    ratio_full = EvaluationEngine.compute_inlier_ratio(inlier_count=80, total_matches=80)
    assert ratio_full == 100.0


def test_projective_rmse_computation():
    """Verifies Sub-pixel Registration RMSE: sqrt(1/N * sum(||x_ref - H * x_src||^2))."""
    # Define source tie points
    src_pts = np.array([
        [10.0, 10.0],
        [50.0, 10.0],
        [10.0, 50.0],
        [50.0, 50.0],
    ], dtype=np.float64)

    # Known homography H: translation dx=3.0, dy=-2.0
    H = np.array([
        [1.0, 0.0, 3.0],
        [0.0, 1.0, -2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # Injected residual offsets
    offsets = np.array([
        [0.1, 0.2],    # norm^2 = 0.01 + 0.04 = 0.05
        [-0.2, 0.1],   # norm^2 = 0.04 + 0.01 = 0.05
        [0.0, -0.3],   # norm^2 = 0.00 + 0.09 = 0.09
        [0.2, -0.1],   # norm^2 = 0.04 + 0.01 = 0.05
    ], dtype=np.float64)
    # Total sum of squares = 0.05 + 0.05 + 0.09 + 0.05 = 0.24
    # Mean square = 0.24 / 4 = 0.06
    # True RMSE = sqrt(0.06) ≈ 0.24494897

    # Reference points are H * src_pts + offsets
    ideal_ref = src_pts + np.array([3.0, -2.0])
    ref_pts = ideal_ref + offsets

    rmse, residuals, reproj_ref = EvaluationEngine.compute_projective_rmse(ref_pts, src_pts, H)

    expected_rmse = float(np.sqrt(0.06))
    assert np.isclose(rmse, expected_rmse, atol=1e-5)
    assert len(residuals) == 4
    assert np.allclose(residuals, np.linalg.norm(offsets, axis=1), atol=1e-5)
    assert rmse < 0.40  # Passes ISRO sub-pixel mandate


def test_spatial_distribution_uniformity_entropy():
    """Verifies 2D Shannon Spatial Entropy quantifies feature clumping vs uniformity."""
    shape = (100, 100)

    # Case A: Fully clumped points (all inside cell 0,0)
    clumped_pts = np.array([[5.0, 5.0] for _ in range(50)])
    entropy_clumped = EvaluationEngine.compute_spatial_entropy(clumped_pts, shape, grid_bins=8)
    assert entropy_clumped == 0.0, "Clumped features must produce zero spatial entropy"

    # Case B: Uniformly distributed grid across all 8x8 cells
    xs = np.linspace(6.25, 93.75, 8)
    ys = np.linspace(6.25, 93.75, 8)
    gx, gy = np.meshgrid(xs, ys)
    uniform_pts = np.column_stack([gx.ravel(), gy.ravel()])

    entropy_uniform = EvaluationEngine.compute_spatial_entropy(uniform_pts, shape, grid_bins=8)
    assert np.isclose(entropy_uniform, 1.0, atol=1e-3), f"Uniform points must yield entropy ≈ 1.0, got {entropy_uniform}"


def test_reprojection_only_results_do_not_count_as_ground_truth_validation():
    """Reprojection-consensus metrics are never treated as scientific validation without independent GT."""
    src_pts = np.array([
        [10.0, 10.0],
        [50.0, 10.0],
        [10.0, 50.0],
        [50.0, 50.0],
    ], dtype=np.float64)
    ref_pts = src_pts + np.array([[0.10, -0.05], [-0.05, 0.20], [0.20, 0.10], [-0.15, -0.15]])
    H = np.eye(3)

    inliers = [
        KeypointMatch(ref_xy=(float(ref_pts[i, 0]), float(ref_pts[i, 1])), target_xy=(float(src_pts[i, 0]), float(src_pts[i, 1])), confidence=0.9)
        for i in range(len(src_pts))
    ]

    report = EvaluationEngine.generate_report(
        total_matches=10,
        inliers=inliers,
        image_shape=(100, 100),
        homography=H,
    )

    assert report.ground_truth_available is False
    assert report.meets_isro_mandate is False
    assert report.metric_basis == "reprojection_consensus"


def test_export_structured_json_and_scatter_plot(tmp_path: Path):
    """Verifies direct export to structured JSON report and residual error scatter plot."""
    # Create sample tie points
    src_pts = np.array([
        [20.0, 20.0],
        [80.0, 20.0],
        [20.0, 80.0],
        [80.0, 80.0],
        [50.0, 50.0],
    ])
    # Slight perturbation with RMSE ≈ 0.22 px
    ref_pts = src_pts + np.array([[0.15, -0.1], [-0.1, 0.2], [0.2, 0.1], [-0.15, -0.15], [0.05, 0.05]])
    H = np.eye(3)

    inliers: list[KeypointMatch] = []
    for i in range(len(src_pts)):
        inliers.append(
            KeypointMatch(
                ref_xy=(float(ref_pts[i, 0]), float(ref_pts[i, 1])),
                target_xy=(float(src_pts[i, 0]), float(src_pts[i, 1])),
                confidence=0.88,
            )
        )

    report = EvaluationEngine.generate_report(
        total_matches=10,
        inliers=inliers,
        image_shape=(100, 100),
        homography=H,
        processing_time_ms=12.5,
    )

    assert report.inlier_count == 5
    assert report.total_matches == 10
    assert report.inlier_ratio_percent == 50.0
    assert report.rmse_pixels < 0.40
    assert report.meets_isro_mandate is False
    assert report.ground_truth_available is False
    assert report.metric_basis == "reprojection_consensus"
    assert report.spatial_uniformity_entropy > 0.0

    # 1. Test JSON Export
    json_path = tmp_path / "evaluation_report.json"
    json_str = report.export_json(json_path)
    assert json_path.exists()
    parsed = json.loads(json_path.read_text(encoding="utf-8"))

    assert parsed["metadata"]["mission"] == "ISRO Chandrayaan-2 Planetary Remote Sensing"
    assert parsed["summary"]["inlier_ratio_percent"] == 50.0
    assert parsed["summary"]["rmse_pixels"] < 0.40
    assert parsed["summary"]["meets_isro_mandate"] is False
    assert parsed["metadata"]["ground_truth_available"] is False
    assert parsed["metadata"]["metric_basis"] == "reprojection_consensus"
    assert len(parsed["tie_points"]) == 5

    # 2. Test Scatter Plot Export
    plot_path = tmp_path / "residual_scatter_plot.png"
    bg = np.random.uniform(0.2, 0.8, (100, 100)).astype(np.float32)
    saved_path = report.export_residual_scatter_plot(plot_path, background_image=bg)

    assert saved_path.exists()
    assert saved_path.stat().st_size > 1000  # Non-trivial image file

    # 3. Test CSV Export
    csv_path = tmp_path / "evaluation_report.csv"
    csv_str = report.export_csv(csv_path)
    assert csv_path.exists()
    assert "SAMANVAYA PLANETARY REGISTRATION EVALUATION REPORT" in csv_str
    assert "rmse_pixels," in csv_str
    assert "inlier_ratio_percent,50.00,%" in csv_str
    assert "TIE POINT RESIDUAL ERROR TABLE" in csv_str


def test_standalone_metrics_module(tmp_path: Path):
    """Verifies the canonical evaluation metrics functions and reports."""
    from lunar_core.evaluation.metrics import (
        compute_control_points_rmse,
        compute_inlier_stats,
        compute_spatial_uniformity_score,
        evaluate_registration,
    )

    # 1. Inlier stats
    inliers, ratio = compute_inlier_stats(total_matches=50, inlier_count=40)
    assert inliers == 40
    assert ratio == 80.0

    # 2. Control points RMSE
    gt_src = np.array([[10.0, 10.0], [50.0, 50.0], [10.0, 50.0], [50.0, 10.0]])
    gt_ref = gt_src + np.array([2.0, -1.0])
    H = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]])
    cp_rmse = compute_control_points_rmse(gt_ref, gt_src, H)
    assert np.isclose(cp_rmse, 0.0, atol=1e-5)

    # 3. Spatial uniformity score
    kpts = np.array([[20.0, 20.0], [80.0, 80.0], [20.0, 80.0], [80.0, 20.0]])
    score = compute_spatial_uniformity_score(kpts, image_shape=(100, 100), grid_bins=4)
    assert score > 0.0

    # 4. End-to-end report generation and exports
    tie_pts = [
        {"ref_x": float(gt_ref[i, 0]), "ref_y": float(gt_ref[i, 1]), "src_x": float(gt_src[i, 0]), "src_y": float(gt_src[i, 1])}
        for i in range(4)
    ]
    report = evaluate_registration(
        tie_points=tie_pts,
        transformation_matrix=H,
        ground_truth_control_points=(gt_ref, gt_src),
        total_matches=5,
        image_shape=(100, 100),
    )

    assert report.meets_isro_mandate is True
    assert report.rmse_pixels < 0.40
    assert report.control_point_rmse_pixels is not None
    assert np.isclose(report.control_point_rmse_pixels, 0.0, atol=1e-5)

    json_f = tmp_path / "eval_standalone.json"
    csv_f = tmp_path / "eval_standalone.csv"
    report.export_json(json_f)
    report.export_csv(csv_f)

    assert json_f.exists()
    assert csv_f.exists()
    assert "meets_isro_mandate" in json_f.read_text()
    assert "TIE POINT RESIDUAL TABLE" in csv_f.read_text()
