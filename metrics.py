#!/usr/bin/env python3
"""
Samanvaya: Autonomous Lunar Optical Image Registration & Correspondence Framework
ISRO SIH PS 26166 — Comprehensive Evaluation Metric Dashboard & Benchmark Runner.

Thin CLI entrypoint wrapping lunar_core.evaluation.metrics.
"""

from __future__ import annotations

from lunar_core.evaluation.metrics import (
    EvaluationEngine,
    EvaluationReport,
    RegistrationEvaluationReport,
    compute_control_points_rmse,
    compute_inlier_stats,
    compute_projective_reprojection,
    compute_spatial_uniformity_score,
    evaluate_registration,
    run_real_evaluation_benchmark,
    run_standalone_evaluation_demo,
    main,
)

__all__ = [
    "EvaluationEngine",
    "EvaluationReport",
    "RegistrationEvaluationReport",
    "compute_control_points_rmse",
    "compute_inlier_stats",
    "compute_projective_reprojection",
    "compute_spatial_uniformity_score",
    "evaluate_registration",
    "run_real_evaluation_benchmark",
    "run_standalone_evaluation_demo",
    "main",
]

if __name__ == "__main__":
    main()
