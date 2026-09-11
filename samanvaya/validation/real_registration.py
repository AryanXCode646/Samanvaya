"""Auditable real-product registration orchestration.

No synthetic fallback is permitted in this module. Synthetic CI fixtures may call
these functions, but their manifest scope must remain synthetic and is never
promoted to real-data evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from lunar_core.data_io.mission_catalog import inspect_product
from lunar_core.models import SunAngles, TransformationType
from lunar_core.pipeline import LunarCorePipeline
from samanvaya.provenance import config_hash, git_commit_sha
from samanvaya.registration.transform import (
    RegistrationTransform,
    check_geometric_plausibility,
    check_spatial_model_mismatch,
)
from samanvaya.registration.windows import extract_registration_windows


@dataclass
class RealRegistrationResult:
    status: str
    source_id: str
    reference_id: str
    source_instrument: Optional[str]
    reference_instrument: Optional[str]
    overlap_status: str
    scale_ratio: Optional[float]
    illumination_metadata: dict[str, Any]
    representation: str
    matcher: str
    raw_match_count: Optional[int]
    spatially_selected_match_count: Optional[int]
    inlier_count: Optional[int]
    inlier_ratio: Optional[float]
    transform_model: str
    transform_parameters: Optional[list[list[float]]]
    subpixel_count: Optional[int]
    coverage_fraction: Optional[float]
    residual_statistics: dict[str, Any]
    registered_output: Optional[str]
    match_point_output: Optional[str]
    validation_status: str
    provenance: dict[str, Any]
    failure_reason: Optional[str] = None
    source_mission: Optional[str] = None
    reference_mission: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _missing_result(pair_id: str, source_id: str, reference_id: str, reason: str, provenance: dict[str, Any], status: str = "DATA_REQUIRED") -> RealRegistrationResult:
    return RealRegistrationResult(
        status=status,
        source_id=source_id,
        reference_id=reference_id,
        source_instrument=None,
        reference_instrument=None,
        overlap_status="NOT_EVALUATED",
        scale_ratio=None,
        illumination_metadata={},
        representation="NOT_RUN",
        matcher="NOT_RUN",
        raw_match_count=None,
        spatially_selected_match_count=None,
        inlier_count=None,
        inlier_ratio=None,
        transform_model="NOT_RUN",
        transform_parameters=None,
        subpixel_count=None,
        coverage_fraction=None,
        residual_statistics={},
        registered_output=None,
        match_point_output=None,
        validation_status=status,
        provenance={**provenance, "pair_id": pair_id},
        failure_reason=reason,
    )


def _read_primary_band(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.open(path) as dataset:
        if dataset.count < 1 or dataset.width < 2 or dataset.height < 2:
            raise ValueError("INVALID_DIMENSIONS")
        data = dataset.read(1, masked=True).astype(np.float32).filled(np.nan)
        if not np.isfinite(data).any():
            raise ValueError("NO_FINITE_PIXEL_VALUES")
        profile = dataset.profile.copy()
        profile.update(count=1, dtype="float32", compress="lzw")
    return data, profile


def _write_registered(path: Path, data: np.ndarray, profile: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = np.nan_to_num(data.astype(np.float32), nan=-9999.0, posinf=-9999.0, neginf=-9999.0)
    profile = dict(profile)
    profile.update(height=clean.shape[0], width=clean.shape[1], nodata=profile.get("nodata", -9999.0))
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(clean, 1)
    with rasterio.open(path) as reopened:
        if reopened.width != clean.shape[1] or reopened.height != clean.shape[0]:
            raise ValueError("OUTPUT_VERIFICATION_FAILED: dimensions changed after writing")
        check = reopened.read(1, masked=True)
        if check.count() == 0:
            raise ValueError("OUTPUT_VERIFICATION_FAILED: output contains no valid pixels")


def _write_matches(path: Path, matches: list[Any], inliers: list[Any], transform: Optional[np.ndarray]) -> None:
    inlier_keys = {
        (round(float(match.target_xy[0]), 6), round(float(match.target_xy[1]), 6), round(float(match.ref_xy[0]), 6), round(float(match.ref_xy[1]), 6))
        for match in inliers
    }
    rows = []
    for index, match in enumerate(matches):
        rows.append({
            "match_id": index,
            "source_x": float(match.target_xy[0]),
            "source_y": float(match.target_xy[1]),
            "reference_x": float(match.ref_xy[0]),
            "reference_y": float(match.ref_xy[1]),
            "confidence": float(match.confidence),
            "method": "pipeline",
            "scale": "metadata",
            "pyramid_level": "unknown",
            "inlier": bool((round(float(match.target_xy[0]), 6), round(float(match.target_xy[1]), 6), round(float(match.ref_xy[0]), 6), round(float(match.ref_xy[1]), 6)) in inlier_keys),
            "residual_px": match.residual_error,
            "subpixel_refined": bool(match.subpixel_refined),
            "subpixel_dx": None,
            "subpixel_dy": None,
            "uncertainty": match.sigma_x,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["match_id"])
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _generate_residual_vector_plot(path: Path, inliers: list[Any], transform_obj: RegistrationTransform) -> None:
    if not inliers:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    src_pts = np.array([m.target_xy for m in inliers], dtype=np.float64)
    ref_pts = np.array([m.ref_xy for m in inliers], dtype=np.float64)
    pred_ref = transform_obj.apply_source_to_reference(src_pts)
    dx = pred_ref[:, 0] - ref_pts[:, 0]
    dy = pred_ref[:, 1] - ref_pts[:, 1]
    ax.scatter(ref_pts[:, 0], ref_pts[:, 1], c="#1f77b4", s=25, label="Reference Inliers")
    ax.quiver(
        ref_pts[:, 0],
        ref_pts[:, 1],
        dx,
        dy,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="#d62728",
        width=0.003,
        label="Residual Vector (dx, dy)",
    )
    ax.set_title("Reprojection Residual Vector Field (FULL_REFERENCE_IMAGE frame)")
    ax.set_xlabel("Reference X (pixels)")
    ax.set_ylabel("Reference Y (pixels)")
    ax.legend(loc="upper right")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def register_products(source_product: Any, reference_product: Any, output_dir: str | Path, *, config: Optional[dict[str, Any]] = None) -> RealRegistrationResult:
    """Register two inspected MissionProduct objects with no synthetic fallback."""
    output_path = Path(output_dir).expanduser().resolve()
    config = config or {"transformation": "homography", "subpixel_refinement": True}
    provenance = {
        "repository_commit": git_commit_sha(),
        "config_hash": config_hash(config),
        "source_sha256": _sha256(Path(source_product.image_path)),
        "reference_sha256": _sha256(Path(reference_product.image_path)),
        "validation_scope": "REAL_REGISTRATION",
    }
    source_id = source_product.product_id or str(source_product.image_path)
    reference_id = reference_product.product_id or str(reference_product.image_path)
    try:
        windows = extract_registration_windows(source_product, reference_product)
        source, reference = windows.source_image, windows.reference_image
        with rasterio.open(Path(reference_product.image_path)) as reference_dataset:
            reference_profile = reference_dataset.profile.copy()
            reference_profile.update(count=1, dtype="float32", compress="lzw")
        provenance["window_audit"] = {
            "source_offset": [windows.source_window.col_off, windows.source_window.row_off],
            "reference_offset": [windows.reference_window.col_off, windows.reference_window.row_off],
            "overlap_status": windows.overlap_status,
            "geometry_method": windows.geometry_method,
        }
    except (OSError, ValueError) as exc:
        return _missing_result("unassigned", source_id, reference_id, str(exc), provenance)

    scale_ratio = None
    if source_product.gsd_m and reference_product.gsd_m and source_product.gsd_m > 0 and reference_product.gsd_m > 0:
        scale_ratio = max(source_product.gsd_m, reference_product.gsd_m) / min(source_product.gsd_m, reference_product.gsd_m)
    source_gsd = source_product.gsd_m
    reference_gsd = reference_product.gsd_m
    if source_gsd is None:
        source_gsd = (abs(windows.source_transform.a) + abs(windows.source_transform.e)) / 2.0
    if reference_gsd is None:
        reference_gsd = (abs(windows.reference_transform.a) + abs(windows.reference_transform.e)) / 2.0
    if not source_gsd or not reference_gsd or source_gsd <= 0 or reference_gsd <= 0:
        return _missing_result("unassigned", source_id, reference_id, "INVALID_METADATA: usable GSD is unavailable", provenance)
    scale_ratio = max(source_gsd, reference_gsd) / min(source_gsd, reference_gsd)
    ref_sun = None
    source_sun = None
    if reference_product.sun_azimuth_deg is not None and reference_product.sun_elevation_deg is not None:
        ref_sun = SunAngles(reference_product.sun_azimuth_deg, reference_product.sun_elevation_deg)
    if source_product.sun_azimuth_deg is not None and source_product.sun_elevation_deg is not None:
        source_sun = SunAngles(source_product.sun_azimuth_deg, source_product.sun_elevation_deg)

    started = time.perf_counter()
    try:
        result = LunarCorePipeline(transformation_type=TransformationType.HOMOGRAPHY).register(
            reference,
            source,
            ref_sun=ref_sun,
            target_sun=source_sun,
            ref_gsd=reference_gsd,
            target_gsd=source_gsd,
        )
    except ValueError as exc:
        if "MIXED_COORDINATE_FRAMES" in str(exc):
            return _missing_result("unassigned", source_id, reference_id, "MIXED_COORDINATE_FRAMES", provenance, status="MIXED_COORDINATE_FRAMES")
        raise

    if len(result.matches) == 0:
        return RealRegistrationResult(
            status="NO_CORRESPONDENCE",
            source_id=source_id,
            reference_id=reference_id,
            source_instrument=source_product.instrument,
            reference_instrument=reference_product.instrument,
            overlap_status=windows.overlap_status,
            scale_ratio=scale_ratio,
            illumination_metadata={"source": source_sun is not None, "reference": ref_sun is not None},
            representation="phase_congruency",
            matcher=result.matcher_path,
            raw_match_count=0,
            spatially_selected_match_count=0,
            inlier_count=0,
            inlier_ratio=0.0,
            transform_model="homography",
            transform_parameters=None,
            subpixel_count=0,
            coverage_fraction=0.0,
            residual_statistics={},
            registered_output=None,
            match_point_output=None,
            validation_status="NO_MATCHES",
            provenance={**provenance, "runtime_ms": (time.perf_counter() - started) * 1000.0},
            failure_reason="NO_MATCHES: No feature matches found.",
            source_mission=source_product.mission,
            reference_mission=reference_product.mission,
        )

    if result.transform_matrix is None or len(result.inliers) < 4:
        return RealRegistrationResult(
            status="NO_CORRESPONDENCE",
            source_id=source_id,
            reference_id=reference_id,
            source_instrument=source_product.instrument,
            reference_instrument=reference_product.instrument,
            overlap_status=windows.overlap_status,
            scale_ratio=scale_ratio,
            illumination_metadata={"source": source_sun is not None, "reference": ref_sun is not None},
            representation="phase_congruency",
            matcher=result.matcher_path,
            raw_match_count=len(result.matches),
            spatially_selected_match_count=len(result.matches),
            inlier_count=len(result.inliers),
            inlier_ratio=float(result.metrics.inlier_ratio),
            transform_model="homography",
            transform_parameters=None,
            subpixel_count=sum(bool(match.subpixel_refined) for match in result.inliers),
            coverage_fraction=None,
            residual_statistics={},
            registered_output=None,
            match_point_output=None,
            validation_status="LOW_INLIER_RATIO",
            provenance={**provenance, "runtime_ms": (time.perf_counter() - started) * 1000.0},
            failure_reason="LOW_INLIER_RATIO: Fewer than four verified inliers or degenerate transform matrix.",
            source_mission=source_product.mission,
            reference_mission=reference_product.mission,
        )

    # Wrap in RegistrationTransform
    reg_transform = RegistrationTransform(
        model_type="HOMOGRAPHY",
        source_frame="FULL_SOURCE_IMAGE",
        target_frame="FULL_REFERENCE_IMAGE",
        parameters=result.transform_matrix,
        estimation_method="USAC_MAGSAC",
        fit_statistics={"inlier_count": len(result.inliers), "inlier_ratio": float(result.metrics.inlier_ratio)},
    )

    # Plausibility Gate
    plausible, plausibility_reason = check_geometric_plausibility(reg_transform)
    if not plausible:
        return RealRegistrationResult(
            status="IMPLAUSIBLE_TRANSFORM",
            source_id=source_id,
            reference_id=reference_id,
            source_instrument=source_product.instrument,
            reference_instrument=reference_product.instrument,
            overlap_status=windows.overlap_status,
            scale_ratio=scale_ratio,
            illumination_metadata={"source": source_sun is not None, "reference": ref_sun is not None},
            representation="phase_congruency",
            matcher=result.matcher_path,
            raw_match_count=len(result.matches),
            spatially_selected_match_count=len(result.matches),
            inlier_count=len(result.inliers),
            inlier_ratio=float(result.metrics.inlier_ratio),
            transform_model="homography",
            transform_parameters=result.transform_matrix.tolist(),
            subpixel_count=sum(bool(match.subpixel_refined) for match in result.inliers),
            coverage_fraction=result.metrics.spatial_uniformity_entropy,
            residual_statistics={"reprojection_rmse_px": result.metrics.rmse_pixels},
            registered_output=None,
            match_point_output=None,
            validation_status="IMPLAUSIBLE_TRANSFORM",
            provenance={**provenance, "runtime_ms": (time.perf_counter() - started) * 1000.0},
            failure_reason=plausibility_reason,
            source_mission=source_product.mission,
            reference_mission=reference_product.mission,
        )

    # Spatial Model Mismatch Check
    ref_pts = np.array([m.ref_xy for m in result.inliers], dtype=np.float64)
    res_vals = np.array([m.residual_error for m in result.inliers if m.residual_error is not None], dtype=np.float64)
    mismatch_ok, mismatch_reason = check_spatial_model_mismatch(ref_pts, res_vals)
    if not mismatch_ok:
        return RealRegistrationResult(
            status="MODEL_MISMATCH",
            source_id=source_id,
            reference_id=reference_id,
            source_instrument=source_product.instrument,
            reference_instrument=reference_product.instrument,
            overlap_status=windows.overlap_status,
            scale_ratio=scale_ratio,
            illumination_metadata={"source": source_sun is not None, "reference": ref_sun is not None},
            representation="phase_congruency",
            matcher=result.matcher_path,
            raw_match_count=len(result.matches),
            spatially_selected_match_count=len(result.matches),
            inlier_count=len(result.inliers),
            inlier_ratio=float(result.metrics.inlier_ratio),
            transform_model="homography",
            transform_parameters=result.transform_matrix.tolist(),
            subpixel_count=sum(bool(match.subpixel_refined) for match in result.inliers),
            coverage_fraction=result.metrics.spatial_uniformity_entropy,
            residual_statistics={"reprojection_rmse_px": result.metrics.rmse_pixels},
            registered_output=None,
            match_point_output=None,
            validation_status="MODEL_MISMATCH",
            provenance={**provenance, "runtime_ms": (time.perf_counter() - started) * 1000.0},
            failure_reason=mismatch_reason,
            source_mission=source_product.mission,
            reference_mission=reference_product.mission,
        )

    output_path.mkdir(parents=True, exist_ok=True)
    registered_path = output_path / "registered_source.tif"

    # Warp the ORIGINAL source raster using authoritative source->reference transform
    try:
        warped_original = cv2.warpPerspective(source, result.transform_matrix, (reference.shape[1], reference.shape[0]))
    except Exception as exc:
        return RealRegistrationResult(
            status="WARP_FAILURE",
            source_id=source_id,
            reference_id=reference_id,
            source_instrument=source_product.instrument,
            reference_instrument=reference_product.instrument,
            overlap_status=windows.overlap_status,
            scale_ratio=scale_ratio,
            illumination_metadata={"source": source_sun is not None, "reference": ref_sun is not None},
            representation="phase_congruency",
            matcher=result.matcher_path,
            raw_match_count=len(result.matches),
            spatially_selected_match_count=len(result.matches),
            inlier_count=len(result.inliers),
            inlier_ratio=float(result.metrics.inlier_ratio),
            transform_model="homography",
            transform_parameters=result.transform_matrix.tolist(),
            subpixel_count=sum(bool(match.subpixel_refined) for match in result.inliers),
            coverage_fraction=result.metrics.spatial_uniformity_entropy,
            residual_statistics={"reprojection_rmse_px": result.metrics.rmse_pixels},
            registered_output=None,
            match_point_output=None,
            validation_status="WARP_FAILURE",
            provenance={**provenance, "runtime_ms": (time.perf_counter() - started) * 1000.0},
            failure_reason=f"Raster warping failed: {exc}",
            source_mission=source_product.mission,
            reference_mission=reference_product.mission,
        )

    try:
        _write_registered(registered_path, warped_original, reference_profile)
    except Exception as exc:
        return RealRegistrationResult(
            status="OUTPUT_VERIFICATION_FAILED",
            source_id=source_id,
            reference_id=reference_id,
            source_instrument=source_product.instrument,
            reference_instrument=reference_product.instrument,
            overlap_status=windows.overlap_status,
            scale_ratio=scale_ratio,
            illumination_metadata={"source": source_sun is not None, "reference": ref_sun is not None},
            representation="phase_congruency",
            matcher=result.matcher_path,
            raw_match_count=len(result.matches),
            spatially_selected_match_count=len(result.matches),
            inlier_count=len(result.inliers),
            inlier_ratio=float(result.metrics.inlier_ratio),
            transform_model="homography",
            transform_parameters=result.transform_matrix.tolist(),
            subpixel_count=sum(bool(match.subpixel_refined) for match in result.inliers),
            coverage_fraction=result.metrics.spatial_uniformity_entropy,
            residual_statistics={"reprojection_rmse_px": result.metrics.rmse_pixels},
            registered_output=None,
            match_point_output=None,
            validation_status="OUTPUT_VERIFICATION_FAILED",
            provenance={**provenance, "runtime_ms": (time.perf_counter() - started) * 1000.0},
            failure_reason=f"Output verification failed: {exc}",
            source_mission=source_product.mission,
            reference_mission=reference_product.mission,
        )

    matches_path = output_path / "matches.csv"
    _write_matches(matches_path, result.matches, result.inliers, result.transform_matrix)
    _write_matches(output_path / "estimation_matches.csv", result.matches, result.inliers, result.transform_matrix)

    # Record output georeferencing validation
    src_transform_repr = [float(x) for x in list(windows.source_transform)[:6]] if hasattr(windows.source_transform, "__iter__") else str(windows.source_transform)
    ref_transform_repr = [float(x) for x in list(windows.reference_transform)[:6]] if hasattr(windows.reference_transform, "__iter__") else str(windows.reference_transform)
    output_validation = {
        "width": int(reference.shape[1]),
        "height": int(reference.shape[0]),
        "dtype": str(reference_profile.get("dtype", "float32")),
        "nodata": float(reference_profile.get("nodata", -9999.0)),
        "crs": str(reference_profile.get("crs", "")),
        "input_source_transform": src_transform_repr,
        "input_reference_transform": ref_transform_repr,
        "registration_transform": result.transform_matrix.tolist(),
    }
    (output_path / "output_validation.json").write_text(json.dumps(output_validation, indent=2), encoding="utf-8")

    # Record geometric diagnostic report
    geometry_report = {
        "model": "GLOBAL_PROJECTIVE",
        "source_frame": "FULL_SOURCE_IMAGE",
        "reference_frame": "FULL_REFERENCE_IMAGE",
        "parameters": result.transform_matrix.tolist(),
        "fit_point_count": len(result.matches),
        "inlier_count": len(result.inliers),
        "inlier_ratio": float(result.metrics.inlier_ratio),
        "fit_residuals": [float(m.residual_error) for m in result.inliers if m.residual_error is not None],
        "held_out_residuals": [],
        "condition_number": reg_transform.condition_number(),
        "plausibility_status": "PLAUSIBLE",
    }
    (output_path / "geometry.json").write_text(json.dumps(geometry_report, indent=2), encoding="utf-8")

    # Generate residual vectors visual artifact
    _generate_residual_vector_plot(output_path / "residual_vectors.png", result.inliers, reg_transform)

    return RealRegistrationResult(
        status="SUCCESS",
        source_id=source_id,
        reference_id=reference_id,
        source_instrument=source_product.instrument,
        reference_instrument=reference_product.instrument,
        overlap_status=windows.overlap_status,
        scale_ratio=scale_ratio,
        illumination_metadata={"source": source_sun is not None, "reference": ref_sun is not None},
        representation="phase_congruency",
        matcher=result.matcher_path,
        raw_match_count=len(result.matches),
        spatially_selected_match_count=len(result.matches),
        inlier_count=len(result.inliers),
        inlier_ratio=float(result.metrics.inlier_ratio),
        transform_model="homography",
        transform_parameters=result.transform_matrix.tolist(),
        subpixel_count=sum(bool(match.subpixel_refined) for match in result.inliers),
        coverage_fraction=result.metrics.spatial_uniformity_entropy,
        residual_statistics={"reprojection_rmse_px": result.metrics.rmse_pixels},
        registered_output=str(registered_path),
        match_point_output=str(matches_path),
        validation_status="REAL_REGISTRATION_PENDING_INDEPENDENT_VALIDATION",
        provenance={**provenance, "runtime_ms": (time.perf_counter() - started) * 1000.0},
        source_mission=source_product.mission,
        reference_mission=reference_product.mission,
    )
