"""Scientific ablation study across 7 modular stages of the Samanvaya pipeline.

Evaluates incremental contribution of each algorithmic stage:
1. raw_classical: Standard classical SIFT/ORB + standard RANSAC (no photometric norm, no multiscale, no ANMS, no subpixel)
2. illumination_norm: Photometric normalization / phase congruency representation
3. multiscale_pyramid: Illumination norm + Coarse-to-fine multiscale pyramid / ROI extraction
4. geometric_filtering: Multiscale + USAC-MAGSAC++ robust estimation
5. anms_distribution: Geometric filtering + 8x8 Grid ANMS spatial uniformization
6. subpixel_refinement: ANMS + Analytical paraboloid sub-pixel refinement
7. full_pipeline: Full integrated Samanvaya pipeline with model selection
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from lunar_core.alignment.dense_matcher import DenseTransformerMatcher
from lunar_core.alignment.rift_matcher import ClassicalRIFTMatcher
from lunar_core.models import KeypointMatch, SunAngles, TransformationType
from lunar_core.alignment.scale_space import ScaleSpaceLocalizer
from lunar_core.pipeline import LunarCorePipeline
from lunar_core.postprocessing.anms import SpatialUniformDistributor
from lunar_core.postprocessing.subpixel import AnalyticalSubpixelRefiner
from lunar_core.preprocessing.contrast import DynamicContrastEqualizer
from lunar_core.preprocessing.phase_congruency import PhaseCongruencyEngine
from lunar_core.preprocessing.photometric import PhotometricNormalizer
from lunar_core.postprocessing.magsac import RobustEstimator
from samanvaya.provenance import git_commit_sha
from samanvaya.registration.transform import RegistrationTransform, check_geometric_plausibility
from samanvaya.validation.baseline_registration import run_baseline_registration
from samanvaya.validation.checkpoints import evaluate_checkpoints


@dataclass
class AblationStageResult:
    stage_index: int
    stage_name: str
    description: str
    status: str
    runtime_ms: float
    raw_match_count: int
    inlier_count: int
    inlier_ratio: float
    reprojection_rmse_px: Optional[float]
    median_residual_px: Optional[float]
    p95_residual_px: Optional[float]
    spatial_entropy: Optional[float]
    coverage_ratio: Optional[float]
    held_out_rmse_px: Optional[float] = None
    transform_matrix: Optional[list[list[float]]] = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _evaluate_residuals(inliers: list[KeypointMatch], H: np.ndarray) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if not inliers or H is None or len(inliers) < 4:
        return None, None, None
    src_pts = np.array([m.target_xy for m in inliers], dtype=np.float64)
    ref_pts = np.array([m.ref_xy for m in inliers], dtype=np.float64)
    src_h = np.column_stack([src_pts, np.ones(len(src_pts))])
    proj = (H @ src_h.T).T
    valid = np.abs(proj[:, 2]) > 1e-8
    if not np.any(valid):
        return None, None, None
    proj_norm = proj[valid, :2] / proj[valid, 2:3]
    ref_valid = ref_pts[valid]
    res = np.linalg.norm(proj_norm - ref_valid, axis=1)
    rmse = float(np.sqrt(np.mean(np.square(res))))
    med = float(np.median(res))
    p95 = float(np.percentile(res, 95))
    return rmse, med, p95


def run_ablation_study(
    source: np.ndarray,
    reference: np.ndarray,
    output_dir: str | Path,
    *,
    source_sun: Optional[SunAngles] = None,
    ref_sun: Optional[SunAngles] = None,
    source_gsd: float = 1.0,
    ref_gsd: float = 1.0,
    checkpoints: Optional[list[dict[str, Any]]] = None,
    pair_id: str = "ablation_study",
) -> dict[str, Any]:
    """Execute all 7 ablation stages sequentially and log comparative benchmarks."""
    out_path = Path(output_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    stages: list[AblationStageResult] = []
    distributor = SpatialUniformDistributor(grid_rows=8, grid_cols=8)

    # -------------------------------------------------------------
    # Stage 1: Raw Classical (SIFT/ORB + standard RANSAC)
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    b_res = run_baseline_registration(source, reference, checkpoints=checkpoints)
    t1 = time.perf_counter()
    stages.append(AblationStageResult(
        stage_index=1,
        stage_name="raw_classical",
        description="Classical SIFT/ORB + Standard OpenCV RANSAC (no photometric norm, no multiscale, no ANMS, no subpixel)",
        status=b_res.get("status", "FAILED"),
        runtime_ms=(t1 - t0) * 1000.0,
        raw_match_count=int(b_res.get("raw_match_count", 0)),
        inlier_count=int(b_res.get("inlier_count", 0)),
        inlier_ratio=float(b_res.get("inlier_ratio", 0.0)),
        reprojection_rmse_px=b_res.get("rmse_pixels"),
        median_residual_px=None,
        p95_residual_px=None,
        spatial_entropy=None,
        coverage_ratio=float(b_res.get("coverage_fraction", 0.0)),
        held_out_rmse_px=b_res.get("held_out_rmse_pixels"),
        transform_matrix=b_res.get("transform_matrix"),
        notes=f"Descriptor: {b_res.get('descriptor', 'SIFT')}",
    ))

    # Shared pre-computations for subsequent stages
    photometric = PhotometricNormalizer()
    pc_engine = PhaseCongruencyEngine(num_scales=3, num_orientations=4)
    dense_matcher = DenseTransformerMatcher()
    estimator = RobustEstimator(threshold_pixels=1.5)
    subpixel_refiner = AnalyticalSubpixelRefiner()

    if source_sun and ref_sun:
        img_ref_norm, _ = photometric.normalize(reference, ref_sun, pixel_gsd=ref_gsd)
        img_tgt_norm, _ = photometric.normalize(source, source_sun, pixel_gsd=source_gsd)
    else:
        img_ref_norm = (reference - np.min(reference)) / (np.ptp(reference) + 1e-6)
        img_tgt_norm = (source - np.min(source)) / (np.ptp(source) + 1e-6)

    pc_ref = pc_engine.compute(img_ref_norm)
    pc_tgt = pc_engine.compute(img_tgt_norm)

    # -------------------------------------------------------------
    # Stage 2: Illumination Norm Only (Phase Congruency + Dense Matcher, Basic RANSAC)
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    try:
        raw_matches_s2 = dense_matcher.match_patches(pc_ref.max_moment, pc_tgt.max_moment)
        key_matches_s2 = [
            KeypointMatch(
                ref_xy=m.ref_xy,
                target_xy=m.target_xy,
                confidence=m.confidence,
                source_frame="FULL_SOURCE_IMAGE",
                reference_frame="FULL_REFERENCE_IMAGE",
            )
            for m in raw_matches_s2
        ]
        if len(key_matches_s2) >= 4:
            src_pts = np.float32([m.target_xy for m in key_matches_s2]).reshape(-1, 1, 2)
            dst_pts = np.float32([m.ref_xy for m in key_matches_s2]).reshape(-1, 1, 2)
            H2, mask2 = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
            inliers_s2 = [m for m, ok in zip(key_matches_s2, mask2.ravel() if mask2 is not None else []) if ok]
            rmse, med, p95 = _evaluate_residuals(inliers_s2, H2)
            st_s2 = "SUCCESS" if len(inliers_s2) >= 4 else "LOW_INLIER_RATIO"
        else:
            H2, inliers_s2, st_s2, rmse, med, p95 = None, [], "NO_MATCHES", None, None, None
    except Exception as exc:
        H2, inliers_s2, st_s2, rmse, med, p95 = None, [], f"ERROR: {exc}", None, None, None
    t1 = time.perf_counter()

    sp_metrics_s2 = distributor.compute_spatial_metrics(inliers_s2, source.shape, use_source_coords=True)
    stages.append(AblationStageResult(
        stage_index=2,
        stage_name="illumination_norm",
        description="Illumination Normalization + Phase Congruency (no multiscale pyramid, standard RANSAC, no ANMS, no subpixel)",
        status=st_s2,
        runtime_ms=(t1 - t0) * 1000.0,
        raw_match_count=len(key_matches_s2) if 'key_matches_s2' in locals() else 0,
        inlier_count=len(inliers_s2),
        inlier_ratio=len(inliers_s2) / max(1, len(key_matches_s2)) if 'key_matches_s2' in locals() else 0.0,
        reprojection_rmse_px=rmse,
        median_residual_px=med,
        p95_residual_px=p95,
        spatial_entropy=sp_metrics_s2["spatial_entropy"],
        coverage_ratio=sp_metrics_s2["coverage_ratio"],
        transform_matrix=H2.tolist() if H2 is not None else None,
    ))

    # -------------------------------------------------------------
    # Stage 3: Multiscale Pyramid (Illum Norm + Coarse-to-fine Fourier-Mellin ROI)
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    roi = ScaleSpaceLocalizer.extract_coarse_roi(pc_ref.max_moment, pc_tgt.max_moment, ref_gsd, source_gsd)
    raw_matches_s3 = dense_matcher.match_patches(roi.ref_roi, roi.target_roi)
    xmin, ymin = roi.ref_bbox[0], roi.ref_bbox[1]
    target_from_common = np.linalg.inv(roi.target_to_common)

    def common_to_target_full(x: float, y: float) -> tuple[float, float]:
        point = target_from_common @ np.array([x + xmin, y + ymin, 1.0], dtype=np.float64)
        point /= point[2]
        return float(point[0] * roi.target_common_to_full_scale), float(point[1] * roi.target_common_to_full_scale)

    def common_to_reference_full(x: float, y: float) -> tuple[float, float]:
        return float((x + xmin) * roi.reference_common_to_full_scale), float((y + ymin) * roi.reference_common_to_full_scale)

    global_matches_s3 = [
        KeypointMatch(
            ref_xy=common_to_reference_full(m.ref_xy[0], m.ref_xy[1]),
            target_xy=common_to_target_full(m.target_xy[0], m.target_xy[1]),
            confidence=m.confidence,
            source_frame="FULL_SOURCE_IMAGE",
            reference_frame="FULL_REFERENCE_IMAGE",
        )
        for m in raw_matches_s3
    ]
    if len(global_matches_s3) >= 4:
        src_pts = np.float32([m.target_xy for m in global_matches_s3]).reshape(-1, 1, 2)
        dst_pts = np.float32([m.ref_xy for m in global_matches_s3]).reshape(-1, 1, 2)
        H3, mask3 = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        inliers_s3 = [m for m, ok in zip(global_matches_s3, mask3.ravel() if mask3 is not None else []) if ok]
        rmse, med, p95 = _evaluate_residuals(inliers_s3, H3)
        st_s3 = "SUCCESS" if len(inliers_s3) >= 4 else "LOW_INLIER_RATIO"
    else:
        H3, inliers_s3, st_s3, rmse, med, p95 = None, [], "NO_MATCHES", None, None, None
    t1 = time.perf_counter()

    sp_metrics_s3 = distributor.compute_spatial_metrics(inliers_s3, source.shape, use_source_coords=True)
    stages.append(AblationStageResult(
        stage_index=3,
        stage_name="multiscale_pyramid",
        description="Illumination Norm + ScaleSpace Fourier-Mellin ROI (Standard RANSAC, no ANMS, no subpixel)",
        status=st_s3,
        runtime_ms=(t1 - t0) * 1000.0,
        raw_match_count=len(global_matches_s3),
        inlier_count=len(inliers_s3),
        inlier_ratio=len(inliers_s3) / max(1, len(global_matches_s3)),
        reprojection_rmse_px=rmse,
        median_residual_px=med,
        p95_residual_px=p95,
        spatial_entropy=sp_metrics_s3["spatial_entropy"],
        coverage_ratio=sp_metrics_s3["coverage_ratio"],
        transform_matrix=H3.tolist() if H3 is not None else None,
    ))

    # -------------------------------------------------------------
    # Stage 4: Geometric Filtering (Stage 3 + USAC-MAGSAC++)
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    H4, inliers_s4 = estimator.estimate(global_matches_s3, TransformationType.HOMOGRAPHY)
    rmse, med, p95 = _evaluate_residuals(inliers_s4, H4)
    st_s4 = "SUCCESS" if len(inliers_s4) >= 4 and H4 is not None else "LOW_INLIER_RATIO"
    t1 = time.perf_counter()

    sp_metrics_s4 = distributor.compute_spatial_metrics(inliers_s4, source.shape, use_source_coords=True)
    stages.append(AblationStageResult(
        stage_index=4,
        stage_name="geometric_filtering",
        description="Multiscale + USAC-MAGSAC++ Robust Geometric Filtering (no ANMS, no subpixel)",
        status=st_s4,
        runtime_ms=(t1 - t0) * 1000.0,
        raw_match_count=len(global_matches_s3),
        inlier_count=len(inliers_s4),
        inlier_ratio=len(inliers_s4) / max(1, len(global_matches_s3)),
        reprojection_rmse_px=rmse,
        median_residual_px=med,
        p95_residual_px=p95,
        spatial_entropy=sp_metrics_s4["spatial_entropy"],
        coverage_ratio=sp_metrics_s4["coverage_ratio"],
        transform_matrix=H4.tolist() if H4 is not None else None,
    ))

    # -------------------------------------------------------------
    # Stage 5: ANMS Spatial Distribution (Stage 4 + 8x8 Grid ANMS)
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    anms_matches_s5 = distributor.cap_grid_cells(global_matches_s3, source.shape, cap_per_cell=4, use_source_coords=True)
    H5, inliers_s5 = estimator.estimate(anms_matches_s5, TransformationType.HOMOGRAPHY)
    rmse, med, p95 = _evaluate_residuals(inliers_s5, H5)
    st_s5 = "SUCCESS" if len(inliers_s5) >= 4 and H5 is not None else "LOW_INLIER_RATIO"
    t1 = time.perf_counter()

    sp_metrics_s5 = distributor.compute_spatial_metrics(inliers_s5, source.shape, use_source_coords=True)
    stages.append(AblationStageResult(
        stage_index=5,
        stage_name="anms_distribution",
        description="USAC-MAGSAC++ + 8x8 Grid ANMS Equal-Cell Spatial Allocation (no subpixel refinement)",
        status=st_s5,
        runtime_ms=(t1 - t0) * 1000.0,
        raw_match_count=len(anms_matches_s5),
        inlier_count=len(inliers_s5),
        inlier_ratio=len(inliers_s5) / max(1, len(anms_matches_s5)),
        reprojection_rmse_px=rmse,
        median_residual_px=med,
        p95_residual_px=p95,
        spatial_entropy=sp_metrics_s5["spatial_entropy"],
        coverage_ratio=sp_metrics_s5["coverage_ratio"],
        transform_matrix=H5.tolist() if H5 is not None else None,
    ))

    # -------------------------------------------------------------
    # Stage 6: Subpixel Refinement (Stage 5 + Analytical Subpixel Refiner)
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    if inliers_s5:
        refined_inliers_s6 = subpixel_refiner.refine_matches_batch(inliers_s5, pc_ref.max_moment, pc_tgt.max_moment, patch_radius=6)
        src_pts = np.array([m.target_xy for m in refined_inliers_s6], dtype=np.float64)
        dst_pts = np.array([m.ref_xy for m in refined_inliers_s6], dtype=np.float64)
        H6, mask6 = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, 1.5)
        inliers_s6 = [m for m, ok in zip(refined_inliers_s6, mask6.ravel() if mask6 is not None else []) if ok]
        rmse, med, p95 = _evaluate_residuals(inliers_s6, H6)
        st_s6 = "SUCCESS" if len(inliers_s6) >= 4 and H6 is not None else "LOW_INLIER_RATIO"
    else:
        H6, inliers_s6, st_s6, rmse, med, p95 = None, [], "NO_INLIERS", None, None, None
    t1 = time.perf_counter()

    sp_metrics_s6 = distributor.compute_spatial_metrics(inliers_s6, source.shape, use_source_coords=True)
    stages.append(AblationStageResult(
        stage_index=6,
        stage_name="subpixel_refinement",
        description="Stage 5 + Analytical Paraboloid Sub-pixel Peak Fitting on Invariant Phase Congruency",
        status=st_s6,
        runtime_ms=(t1 - t0) * 1000.0,
        raw_match_count=len(anms_matches_s5),
        inlier_count=len(inliers_s6),
        inlier_ratio=len(inliers_s6) / max(1, len(anms_matches_s5)),
        reprojection_rmse_px=rmse,
        median_residual_px=med,
        p95_residual_px=p95,
        spatial_entropy=sp_metrics_s6["spatial_entropy"],
        coverage_ratio=sp_metrics_s6["coverage_ratio"],
        transform_matrix=H6.tolist() if H6 is not None else None,
    ))

    # -------------------------------------------------------------
    # Stage 7: Full Pipeline (Integrated LunarCorePipeline)
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    pipeline = LunarCorePipeline(
        transformation_type=TransformationType.HOMOGRAPHY,
        enable_photometric=True,
        enable_anms=True,
        enable_subpixel=True,
    )
    full_result = pipeline.register(
        reference,
        source,
        ref_sun=ref_sun,
        target_sun=source_sun,
        ref_gsd=ref_gsd,
        target_gsd=source_gsd,
    )
    t1 = time.perf_counter()
    H7 = full_result.transform_matrix
    inliers_s7 = full_result.inliers
    rmse, med, p95 = _evaluate_residuals(inliers_s7, H7)
    st_s7 = "SUCCESS" if len(inliers_s7) >= 4 and H7 is not None else "FAILED"
    sp_metrics_s7 = distributor.compute_spatial_metrics(inliers_s7, source.shape, use_source_coords=True)

    stages.append(AblationStageResult(
        stage_index=7,
        stage_name="full_pipeline",
        description="Integrated Samanvaya Production Pipeline (All stages combined + dynamic fallback)",
        status=st_s7,
        runtime_ms=(t1 - t0) * 1000.0,
        raw_match_count=len(full_result.matches),
        inlier_count=len(inliers_s7),
        inlier_ratio=float(full_result.metrics.inlier_ratio),
        reprojection_rmse_px=rmse,
        median_residual_px=med,
        p95_residual_px=p95,
        spatial_entropy=sp_metrics_s7["spatial_entropy"],
        coverage_ratio=sp_metrics_s7["coverage_ratio"],
        transform_matrix=H7.tolist() if H7 is not None else None,
        notes=f"Matcher: {full_result.matcher_path}",
    ))

    # Evaluate held-out checkpoints for all stages if provided
    if checkpoints:
        for stage in stages:
            if stage.transform_matrix is not None:
                chk_res = evaluate_checkpoints(stage.transform_matrix, checkpoints)
                stage.held_out_rmse_px = chk_res.get("rmse_pixels")

    # Serialize JSON
    summary_payload = {
        "pair_id": pair_id,
        "provenance": {
            "repository_commit": git_commit_sha(),
            "evaluation_type": "SCIENTIFIC_ABLATION_STUDY",
        },
        "stages": [s.to_dict() for s in stages],
    }
    (out_path / "ablation_results.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    # Serialize CSV
    csv_fields = [
        "stage_index",
        "stage_name",
        "status",
        "runtime_ms",
        "raw_match_count",
        "inlier_count",
        "inlier_ratio",
        "reprojection_rmse_px",
        "median_residual_px",
        "p95_residual_px",
        "spatial_entropy",
        "coverage_ratio",
        "held_out_rmse_px",
        "notes",
    ]
    with (out_path / "ablation_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for s in stages:
            row = s.to_dict()
            writer.writerow({k: row.get(k) for k in csv_fields})

    return summary_payload
