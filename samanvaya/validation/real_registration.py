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

    @property
    def source_product_id(self) -> str:
        return self.source_id

    @property
    def reference_product_id(self) -> str:
        return self.reference_id

    @property
    def selected_match_count(self) -> Optional[int]:
        return self.spatially_selected_match_count

    @property
    def transform(self) -> Optional[list[list[float]]]:
        return self.transform_parameters

    @property
    def subpixel_statistics(self) -> dict[str, Any]:
        return {
            "subpixel_count": self.subpixel_count,
            "residual_statistics": self.residual_statistics,
        }

    @property
    def match_output(self) -> Optional[str]:
        return self.match_point_output

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["source_product_id"] = self.source_product_id
        d["reference_product_id"] = self.reference_product_id
        d["selected_match_count"] = self.selected_match_count
        d["transform"] = self.transform
        d["subpixel_statistics"] = self.subpixel_statistics
        d["match_output"] = self.match_output
        return d


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


def _generate_8panel_diagnostic_plot(
    path: Path,
    source: np.ndarray,
    reference: np.ndarray,
    warped: np.ndarray,
    matches: list[Any],
    inliers: list[Any],
    reg_transform: RegistrationTransform,
    grid_rows: int = 8,
    grid_cols: int = 8,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=120)

    def _norm(img: np.ndarray) -> np.ndarray:
        fin = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        mn, mx = float(np.min(fin)), float(np.max(fin))
        if mx - mn < 1e-6:
            return np.zeros(fin.shape, dtype=np.float32)
        return (fin - mn) / (mx - mn)

    src_n = _norm(source)
    ref_n = _norm(reference)
    warp_n = _norm(warped)

    # Panel 1: Source
    axes[0, 0].imshow(src_n, cmap="gray")
    axes[0, 0].set_title(f"1. Source Image ({source.shape[1]}x{source.shape[0]})", fontsize=10, fontweight="bold")
    axes[0, 0].set_xlabel("X (px)")
    axes[0, 0].set_ylabel("Y (px)")

    # Panel 2: Reference
    axes[0, 1].imshow(ref_n, cmap="gray")
    axes[0, 1].set_title(f"2. Reference Image ({reference.shape[1]}x{reference.shape[0]})", fontsize=10, fontweight="bold")
    axes[0, 1].set_xlabel("X (px)")
    axes[0, 1].set_ylabel("Y (px)")

    # Panel 3: Raw matches
    axes[0, 2].imshow(src_n, cmap="gray")
    if matches:
        raw_x = [m.target_xy[0] for m in matches[:1000]]
        raw_y = [m.target_xy[1] for m in matches[:1000]]
        axes[0, 2].scatter(raw_x, raw_y, c="cyan", s=8, alpha=0.6, label=f"Raw ({len(matches)})")
        axes[0, 2].legend(loc="upper right", fontsize=8)
    axes[0, 2].set_title(f"3. Raw Candidate Matches (N={len(matches)})", fontsize=10, fontweight="bold")

    # Panel 4: Filtered inliers (RANSAC verified)
    axes[0, 3].imshow(src_n, cmap="gray")
    if inliers:
        inl_x = [m.target_xy[0] for m in inliers]
        inl_y = [m.target_xy[1] for m in inliers]
        axes[0, 3].scatter(inl_x, inl_y, c="lime", s=14, alpha=0.8, label=f"Inliers ({len(inliers)})")
        axes[0, 3].legend(loc="upper right", fontsize=8)
    axes[0, 3].set_title(f"4. Filtered Inliers (N={len(inliers)})", fontsize=10, fontweight="bold")

    # Panel 5: Refined inliers (subpixel refined)
    axes[1, 0].imshow(ref_n, cmap="gray")
    refined = [m for m in inliers if getattr(m, "subpixel_refined", False)]
    if refined:
        ref_rx = [m.ref_xy[0] for m in refined]
        ref_ry = [m.ref_xy[1] for m in refined]
        axes[1, 0].scatter(ref_rx, ref_ry, c="yellow", s=14, alpha=0.8, label=f"Sub-pixel ({len(refined)})")
        axes[1, 0].legend(loc="upper right", fontsize=8)
    elif inliers:
        ref_rx = [m.ref_xy[0] for m in inliers]
        ref_ry = [m.ref_xy[1] for m in inliers]
        axes[1, 0].scatter(ref_rx, ref_ry, c="orange", s=14, alpha=0.8, label=f"Inliers ({len(inliers)})")
        axes[1, 0].legend(loc="upper right", fontsize=8)
    axes[1, 0].set_title(f"5. Refined Inliers on Reference (N={len(refined)})", fontsize=10, fontweight="bold")

    # Panel 6: 8x8 Spatial grid occupancy heatmap
    from lunar_core.postprocessing.anms import SpatialUniformDistributor
    distributor = SpatialUniformDistributor(grid_rows=grid_rows, grid_cols=grid_cols)
    metrics_dist = distributor.compute_spatial_metrics(inliers, source.shape, use_source_coords=True)
    counts_arr = np.array(metrics_dist["points_per_cell"]["counts"], dtype=np.float64).reshape(grid_rows, grid_cols)
    h_src, w_src = source.shape[:2]
    im_heat = axes[1, 1].imshow(counts_arr, extent=[0, w_src, h_src, 0], cmap="viridis", aspect="auto", alpha=0.8)
    if inliers:
        axes[1, 1].scatter([m.target_xy[0] for m in inliers], [m.target_xy[1] for m in inliers], c="white", s=8, alpha=0.7)
    for r in range(grid_rows + 1):
        axes[1, 1].axhline(r * (h_src / grid_rows), color="white", linestyle=":", linewidth=0.5, alpha=0.5)
    for c in range(grid_cols + 1):
        axes[1, 1].axvline(c * (w_src / grid_cols), color="white", linestyle=":", linewidth=0.5, alpha=0.5)
    axes[1, 1].set_title(f"6. 8x8 Spatial Grid (Cov: {metrics_dist['coverage_fraction']:.1%}, H: {metrics_dist['spatial_entropy']:.2f})", fontsize=10, fontweight="bold")
    plt.colorbar(im_heat, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Panel 7: Registered Overlay
    blend = 0.5 * ref_n + 0.5 * warp_n
    axes[1, 2].imshow(blend, cmap="magma")
    axes[1, 2].set_title("7. Registered Overlay (Ref + Warped)", fontsize=10, fontweight="bold")

    # Panel 8: Residual Vectors & Histogram
    residuals = [float(m.residual_error) for m in inliers if getattr(m, "residual_error", None) is not None]
    if residuals:
        n_bins = min(25, max(5, len(residuals) // 2))
        axes[1, 3].hist(residuals, bins=n_bins, color="coral", edgecolor="black", alpha=0.7)
        rmse = float(np.sqrt(np.mean(np.square(residuals))))
        med = float(np.median(residuals))
        p95 = float(np.percentile(residuals, 95))
        axes[1, 3].axvline(rmse, color="red", linestyle="--", linewidth=1.5, label=f"RMSE: {rmse:.2f}px")
        axes[1, 3].axvline(med, color="blue", linestyle=":", linewidth=1.5, label=f"Median: {med:.2f}px")
        axes[1, 3].axvline(p95, color="purple", linestyle="-.", linewidth=1.5, label=f"P95: {p95:.2f}px")
        axes[1, 3].legend(loc="upper right", fontsize=8)
        axes[1, 3].set_xlabel("Residual Magnitude (pixels)")
        axes[1, 3].set_ylabel("Count")
    else:
        axes[1, 3].text(0.5, 0.5, "No residuals computed", ha="center", va="center", transform=axes[1, 3].transAxes)
    axes[1, 3].set_title("8. Reprojection Residual Histogram", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _generate_visual_artifacts(
    visual_dir: Path,
    source: np.ndarray,
    reference: np.ndarray,
    warped: np.ndarray,
    matches: list[Any],
    inliers: list[Any],
    reg_transform: RegistrationTransform,
    selected_matches: list[Any] | None = None,
) -> None:
    visual_dir.mkdir(parents=True, exist_ok=True)

    def _normalize_u8(img: np.ndarray) -> np.ndarray:
        fin = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        mn, mx = float(np.min(fin)), float(np.max(fin))
        if mx - mn < 1e-6:
            return np.zeros(fin.shape, dtype=np.uint8)
        return np.clip((fin - mn) / (mx - mn) * 255.0, 0, 255).astype(np.uint8)

    src_u8 = _normalize_u8(source)
    ref_u8 = _normalize_u8(reference)
    warp_u8 = _normalize_u8(warped)

    # 1. source.png & reference.png
    cv2.imwrite(str(visual_dir / "source.png"), src_u8)
    cv2.imwrite(str(visual_dir / "reference.png"), ref_u8)

    # 2. raw_matches.png, selected_matches.png & verified_matches.png
    def _draw_matches(pts_src: list[Any], pts_ref: list[Any], out_path: Path) -> None:
        h1, w1 = src_u8.shape[:2]
        h2, w2 = ref_u8.shape[:2]
        canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        canvas[:h1, :w1] = cv2.cvtColor(src_u8, cv2.COLOR_GRAY2BGR)
        canvas[:h2, w1:w1 + w2] = cv2.cvtColor(ref_u8, cv2.COLOR_GRAY2BGR)
        for ps, pr in zip(pts_src[:500], pts_ref[:500]):
            pt1 = (int(round(float(ps[0]))), int(round(float(ps[1]))))
            pt2 = (int(round(float(pr[0]))) + w1, int(round(float(pr[1]))))
            cv2.line(canvas, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(canvas, pt1, 2, (0, 0, 255), -1)
            cv2.circle(canvas, pt2, 2, (255, 0, 0), -1)
        cv2.imwrite(str(out_path), canvas)

    if matches:
        _draw_matches([m.target_xy for m in matches], [m.ref_xy for m in matches], visual_dir / "raw_matches.png")
    if selected_matches:
        _draw_matches([m.target_xy for m in selected_matches], [m.ref_xy for m in selected_matches], visual_dir / "selected_matches.png")
    if inliers:
        _draw_matches([m.target_xy for m in inliers], [m.ref_xy for m in inliers], visual_dir / "verified_matches.png")

    # 3. uniform_matches.png (SIH PS 26166: Primary spatial grid on FULL_SOURCE_IMAGE)
    fig, ax = plt.subplots(figsize=(6, 6))
    if inliers:
        src_x = [float(m.target_xy[0]) for m in inliers]
        src_y = [float(m.target_xy[1]) for m in inliers]
        ax.scatter(src_x, src_y, c="#1f77b4", s=18, alpha=0.75, edgecolors="none")
    ax.set_xlim(0, max(1, source.shape[1]))
    ax.set_ylim(max(1, source.shape[0]), 0)
    ax.set_title("Inlier Spatial Distribution (FULL_SOURCE_IMAGE Frame)")
    ax.set_xlabel("Source X / Column (pixels)")
    ax.set_ylabel("Source Y / Row (pixels)")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(visual_dir / "uniform_matches.png", dpi=100)
    fig.savefig(visual_dir / "match_distribution.png", dpi=100)
    plt.close(fig)

    from lunar_core.postprocessing.anms import generate_spatial_distribution_plot
    generate_spatial_distribution_plot(visual_dir / "spatial_distribution_heatmap.png", inliers, source.shape, use_source_coords=True)

    # 4. registered_overlay.png & overlay.png
    blend = cv2.addWeighted(ref_u8, 0.5, warp_u8, 0.5, 0)
    cv2.imwrite(str(visual_dir / "registered_overlay.png"), blend)
    cv2.imwrite(str(visual_dir / "overlay.png"), blend)

    # 5. residual_vectors.png
    _generate_residual_vector_plot(visual_dir / "residual_vectors.png", inliers, reg_transform)

    # 6. diagnostic_dashboard.png (Publication-grade 8-panel diagnostic)
    _generate_8panel_diagnostic_plot(visual_dir / "diagnostic_dashboard.png", source, reference, warped, matches, inliers, reg_transform)


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
    spectral_method = config.get("spectral_representation", "auto") if config else "auto"
    try:
        windows = extract_registration_windows(source_product, reference_product, spectral_method=spectral_method)
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
        err_msg = str(exc)
        status_code = "IIRS_REPRESENTATION_UNCERTAIN" if "IIRS_REPRESENTATION_UNCERTAIN" in err_msg else "DATA_REQUIRED"
        return _missing_result("unassigned", source_id, reference_id, err_msg, provenance, status=status_code)

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

    # Phase 14: Cross-modal classification
    src_inst = str(getattr(source_product, "instrument", "") or "").upper()
    ref_inst = str(getattr(reference_product, "instrument", "") or "").upper()
    src_miss = str(getattr(source_product, "mission", "") or "").upper()
    ref_miss = str(getattr(reference_product, "mission", "") or "").upper()

    if src_inst == "IIRS":
        modality_class = "SPECTRAL_TO_2D"
    elif src_inst == ref_inst and src_miss == ref_miss:
        modality_class = "SAME_MODALITY"
    elif scale_ratio is not None and scale_ratio > 3.0:
        modality_class = "MULTIRESOLUTION"
    else:
        modality_class = "CROSS_MODALITY"

    provenance["modality_classification"] = modality_class
    provenance["spectral_representation"] = windows.spectral_representation
    if windows.spectral_provenance:
        provenance["spectral_provenance"] = windows.spectral_provenance

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
        "nodata": float(reference_profile["nodata"]) if reference_profile.get("nodata") is not None else -9999.0,
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

    # Write complete real output package
    src_meta = source_product.to_dict() if hasattr(source_product, "to_dict") else dict(source_product)
    ref_meta = reference_product.to_dict() if hasattr(reference_product, "to_dict") else dict(reference_product)
    (output_path / "source_metadata.json").write_text(json.dumps(src_meta, indent=2, default=str), encoding="utf-8")
    (output_path / "reference_metadata.json").write_text(json.dumps(ref_meta, indent=2, default=str), encoding="utf-8")

    overlap_info = {
        "overlap_status": windows.overlap_status,
        "geometry_method": windows.geometry_method,
        "scale_ratio": scale_ratio,
        "source_window": {"col_off": windows.source_window.col_off, "row_off": windows.source_window.row_off, "width": windows.source_window.width, "height": windows.source_window.height},
        "reference_window": {"col_off": windows.reference_window.col_off, "row_off": windows.reference_window.row_off, "width": windows.reference_window.width, "height": windows.reference_window.height},
    }
    (output_path / "overlap.json").write_text(json.dumps(overlap_info, indent=2), encoding="utf-8")

    prov_record = {**provenance, "pair_id": output_path.name, "runtime_ms": (time.perf_counter() - started) * 1000.0}
    (output_path / "provenance.json").write_text(json.dumps(prov_record, indent=2, default=str), encoding="utf-8")

    coord_audit = {
        "source_frame": "FULL_SOURCE_IMAGE",
        "reference_frame": "FULL_REFERENCE_IMAGE",
        "transform_direction": "SOURCE_TO_REFERENCE",
        "convention": "T(source FULL_IMAGE) = reference FULL_IMAGE",
        "source_offset": [windows.source_window.col_off, windows.source_window.row_off],
        "reference_offset": [windows.reference_window.col_off, windows.reference_window.row_off],
        "source_gsd_m": source_gsd,
        "reference_gsd_m": reference_gsd,
        "scale_ratio": scale_ratio,
    }
    (output_path / "coordinate_audit.json").write_text(json.dumps(coord_audit, indent=2), encoding="utf-8")

    from lunar_core.postprocessing.anms import SpatialUniformDistributor
    distributor = SpatialUniformDistributor(grid_rows=8, grid_cols=8)
    spatial_dist = distributor.compute_spatial_metrics(result.inliers, source.shape, use_source_coords=True)
    spatial_dist.update({
        "frame": "FULL_SOURCE_IMAGE",
        "raw_match_count": len(result.matches),
        "inlier_count": len(result.inliers),
        "inlier_ratio": float(result.metrics.inlier_ratio),
        "spatial_uniformity_entropy": spatial_dist.get("spatial_entropy", result.metrics.spatial_uniformity_entropy),
    })
    (output_path / "spatial_distribution.json").write_text(json.dumps(spatial_dist, indent=2), encoding="utf-8")

    # Generate visual artifacts
    _generate_residual_vector_plot(output_path / "residual_vectors.png", result.inliers, reg_transform)
    _generate_visual_artifacts(output_path / "visual", source, reference, warped_original, result.matches, result.inliers, reg_transform, selected_matches=result.matches)

    # Standard Section 22 Artifact Exports
    (output_path / "transform.json").write_text(json.dumps({
        "model": "HOMOGRAPHY",
        "matrix": result.transform_matrix.tolist(),
        "source_frame": "FULL_SOURCE_IMAGE",
        "target_frame": "FULL_REFERENCE_IMAGE",
    }, indent=2), encoding="utf-8")

    metrics_record = {
        "rmse_pixels": result.metrics.rmse_pixels,
        "inlier_ratio": float(result.metrics.inlier_ratio),
        "inlier_count": len(result.inliers),
        "raw_matches": len(result.matches),
        "spatial_entropy": spatial_dist.get("spatial_entropy"),
        "coverage_ratio": spatial_dist.get("coverage_ratio"),
        "estimated_transform": result.transform_matrix.tolist(),
    }
    (output_path / "metrics.json").write_text(json.dumps(metrics_record, indent=2), encoding="utf-8")

    import shutil
    if (output_path / "visual" / "overlay.png").exists():
        shutil.copy(output_path / "visual" / "overlay.png", output_path / "overlay.png")
    if (output_path / "visual" / "match_distribution.png").exists():
        shutil.copy(output_path / "visual" / "match_distribution.png", output_path / "match_distribution.png")
    if (output_path / "visual" / "diagnostic_dashboard.png").exists():
        shutil.copy(output_path / "visual" / "diagnostic_dashboard.png", output_path / "diagnostic_dashboard.png")
    if (output_path / "registered_source.tif").exists():
        try:
            (output_path / "registered.tif").symlink_to("registered_source.tif")
        except Exception:
            shutil.copy(output_path / "registered_source.tif", output_path / "registered.tif")

    return RealRegistrationResult(
        status="SUCCESS",
        source_id=source_id,
        reference_id=reference_id,
        source_instrument=source_product.instrument,
        reference_instrument=reference_product.instrument,
        overlap_status=windows.overlap_status,
        scale_ratio=scale_ratio,
        illumination_metadata={"source": source_sun is not None, "reference": ref_sun is not None},
        representation=windows.spectral_representation,
        matcher=result.matcher_path,
        raw_match_count=len(result.matches),
        spatially_selected_match_count=len(result.matches),
        inlier_count=len(result.inliers),
        inlier_ratio=float(result.metrics.inlier_ratio),
        transform_model="homography",
        transform_parameters=result.transform_matrix.tolist(),
        subpixel_count=sum(bool(match.subpixel_refined) for match in result.inliers),
        coverage_fraction=spatial_dist["coverage_fraction"],
        residual_statistics={"reprojection_rmse_px": result.metrics.rmse_pixels},
        registered_output=str(registered_path),
        match_point_output=str(matches_path),
        validation_status="REAL_REGISTRATION_PENDING_INDEPENDENT_VALIDATION",
        provenance=prov_record,
        source_mission=source_product.mission,
        reference_mission=reference_product.mission,
    )


def register_pair(
    source_product: Any,
    reference_product: Any,
    config: Optional[dict[str, Any]] = None,
    output_dir: str | Path = "output/real",
) -> RealRegistrationResult:
    """Authoritative API entrypoint to register two validated MissionProduct objects."""
    return register_products(source_product, reference_product, output_dir=output_dir, config=config)
