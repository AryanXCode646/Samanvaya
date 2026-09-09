"""Library-owned registration runner used by evaluation benchmarks."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from lunar_core.models import SunAngles, TransformationType
from lunar_core.pipeline import LunarCorePipeline
from lunar_core.evaluation.metrics import RegistrationEvaluationReport, evaluate_registration


def run_registration_pipeline(
    src_image: np.ndarray,
    ref_image: np.ndarray,
    src_sun: Optional[SunAngles] = None,
    ref_sun: Optional[SunAngles] = None,
    dem_data: Optional[np.ndarray] = None,
    photometric_mode: str = "minnaert",
    minnaert_k: float = 0.80,
    transformation_model: str = "affine",
    confidence_threshold: float = 0.10,
    ground_truth_control_points: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    output_json: str = "evaluation_report.json",
    output_csv: str = "evaluation_report.csv",
    output_scatter: Optional[str] = None,
    output_warped: Optional[str] = None,
) -> RegistrationEvaluationReport:
    """Run the package pipeline and export its canonical evaluation report."""
    del confidence_threshold, minnaert_k, output_scatter, output_warped
    start = time.perf_counter()
    pipeline = LunarCorePipeline(
        transformation_type=(
            TransformationType.HOMOGRAPHY
            if transformation_model.lower() == "homography"
            else TransformationType.AFFINE
        ),
        enable_photometric=photometric_mode != "none",
        enable_anms=True,
        enable_subpixel=True,
    )
    result = pipeline.register(
        ref_image=ref_image,
        target_image=src_image,
        ref_sun=ref_sun,
        target_sun=src_sun,
        dem_data=dem_data,
    )
    tie_points = [
        {
            "id": index,
            "ref_x": float(match.ref_xy[0]),
            "ref_y": float(match.ref_xy[1]),
            "src_x": float(match.target_xy[0]),
            "src_y": float(match.target_xy[1]),
            "confidence": float(match.confidence),
            "subpixel_refined": bool(match.subpixel_refined),
        }
        for index, match in enumerate(result.inliers)
    ]
    report = evaluate_registration(
        tie_points=tie_points,
        transformation_matrix=result.transform_matrix,
        ground_truth_control_points=ground_truth_control_points,
        total_matches=len(result.matches),
        image_shape=ref_image.shape,
        processing_time_ms=(time.perf_counter() - start) * 1000.0,
    )
    report.export_json(output_json)
    report.export_csv(output_csv)
    return report
