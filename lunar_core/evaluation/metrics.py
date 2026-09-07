"""
Planetary Registration Evaluation Engine: SIH PS 26166 Metrics and Diagnostics.

Computes:
1. Inlier Ratio (%) = (Inlier Count / Total Matches) * 100
2. Sub-pixel Registration RMSE against verified projective tie points:
       RMSE = sqrt( 1/N * sum( ||x_ref - H * x_src||^2 ) )
3. Spatial Distribution Uniformity:
       2D Shannon Spatial Entropy across image grid cells to quantify non-clumping.

Exports directly to:
- Structured JSON evaluation report
- Residual error scatter plot visualization
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np

from lunar_core.models import KeypointMatch, RegistrationMetrics


@dataclass
class RegistrationEvaluationReport:
    """
    Comprehensive evaluation report for planetary image registration.
    Directly exportable to structured JSON and scatter plot diagnostics.
    """
    total_matches: int
    inlier_count: int
    inlier_ratio_percent: float
    rmse_pixels: float
    spatial_uniformity_entropy: float
    mean_residual_pixels: float
    median_residual_pixels: float
    max_residual_pixels: float
    std_residual_pixels: float
    ce90_pixels: float                      # Circular Error at 90th percentile
    meets_isro_mandate: bool                # RMSE < 0.40 px with >= 4 inliers
    control_point_rmse_pixels: Optional[float] = None
    processing_time_ms: float = 0.0
    homography_matrix: Optional[List[List[float]]] = None
    tie_points: List[Dict[str, Any]] = field(default_factory=list)
    image_shape: Tuple[int, int] = (0, 0)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def inlier_ratio(self) -> float:
        """Returns inlier ratio in [0, 1]."""
        return self.inlier_ratio_percent / 100.0

    @property
    def metrics(self) -> RegistrationMetrics:
        """Returns standard RegistrationMetrics instance for backwards compatibility."""
        return RegistrationMetrics(
            rmse_pixels=self.rmse_pixels,
            total_matches=self.total_matches,
            inlier_count=self.inlier_count,
            inlier_ratio=self.inlier_ratio_percent / 100.0,
            spatial_uniformity_entropy=self.spatial_uniformity_entropy,
            mean_residual_pixels=self.mean_residual_pixels,
            max_residual_pixels=self.max_residual_pixels,
            processing_time_ms=self.processing_time_ms,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the report to a Python dictionary."""
        return {
            "metadata": {
                "mission": "ISRO Chandrayaan-2 Planetary Remote Sensing",
                "problem_statement": "SIH PS 26166",
                "timestamp_utc": self.timestamp,
                "framework": "lunar_core v1.0.0",
            },
            "summary": {
                "total_matches": self.total_matches,
                "inlier_count": self.inlier_count,
                "inlier_ratio_percent": round(self.inlier_ratio_percent, 2),
                "rmse_pixels": round(self.rmse_pixels, 4),
                "control_point_rmse_pixels": round(self.control_point_rmse_pixels, 4) if self.control_point_rmse_pixels is not None else None,
                "spatial_uniformity_entropy": round(self.spatial_uniformity_entropy, 4),
                "mean_residual_pixels": round(self.mean_residual_pixels, 4),
                "median_residual_pixels": round(self.median_residual_pixels, 4),
                "max_residual_pixels": round(self.max_residual_pixels, 4),
                "std_residual_pixels": round(self.std_residual_pixels, 4),
                "ce90_pixels": round(self.ce90_pixels, 4),
                "meets_isro_mandate": self.meets_isro_mandate,
                "isro_mandate_threshold_px": 0.40,
                "processing_time_ms": round(self.processing_time_ms, 2),
                "image_shape_hw": list(self.image_shape),
            },
            "homography_matrix": self.homography_matrix,
            "tie_points_count": len(self.tie_points),
            "tie_points": self.tie_points,
        }

    def export_json(self, output_path: Union[str, Path], indent: int = 2) -> str:
        """
        Exports the structured evaluation report to a JSON file.
        """
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        json_str = json.dumps(data, indent=indent)
        out_file.write_text(json_str, encoding="utf-8")
        return json_str

    def export_csv(self, output_path: Union[str, Path]) -> str:
        """
        Exports the evaluation summary and inlier tie point residuals to a CSV file.
        """
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        lines: List[str] = [
            "# SAMANVAYA PLANETARY REGISTRATION EVALUATION REPORT",
            "# Metric,Value,Unit",
            f"total_matches,{self.total_matches},count",
            f"inlier_count,{self.inlier_count},count",
            f"inlier_ratio_percent,{self.inlier_ratio_percent:.2f},%",
            f"rmse_pixels,{self.rmse_pixels:.4f},pixels",
            f"control_point_rmse_pixels,{f'{self.control_point_rmse_pixels:.4f}' if self.control_point_rmse_pixels is not None else 'N/A'},pixels",
            f"spatial_uniformity_entropy,{self.spatial_uniformity_entropy:.4f},normalized_shannon",
            f"mean_residual_pixels,{self.mean_residual_pixels:.4f},pixels",
            f"median_residual_pixels,{self.median_residual_pixels:.4f},pixels",
            f"max_residual_pixels,{self.max_residual_pixels:.4f},pixels",
            f"std_residual_pixels,{self.std_residual_pixels:.4f},pixels",
            f"ce90_pixels,{self.ce90_pixels:.4f},pixels",
            f"meets_isro_mandate,{self.meets_isro_mandate},boolean",
            f"processing_time_ms,{self.processing_time_ms:.2f},ms",
            "#",
            "# TIE POINT RESIDUAL ERROR TABLE",
            "id,ref_x,ref_y,src_x,src_y,reprojected_ref_x,reprojected_ref_y,residual_pixels,confidence,subpixel_refined",
        ]
        for pt in self.tie_points:
            lines.append(
                f"{pt.get('id', 0)},{pt.get('ref_x', 0.0):.3f},{pt.get('ref_y', 0.0):.3f},"
                f"{pt.get('src_x', 0.0):.3f},{pt.get('src_y', 0.0):.3f},"
                f"{pt.get('reprojected_ref_x', 0.0):.3f},{pt.get('reprojected_ref_y', 0.0):.3f},"
                f"{pt.get('residual_pixels', 0.0):.4f},{pt.get('confidence', 1.0):.4f},{pt.get('subpixel_refined', False)}"
            )
        csv_str = "\n".join(lines) + "\n"
        out_file.write_text(csv_str, encoding="utf-8")
        return csv_str

    def export_residual_scatter_plot(
        self,
        output_path: Union[str, Path],
        background_image: Optional[np.ndarray] = None,
        dpi: int = 200,
    ) -> Path:
        """
        Exports a publication-quality residual error scatter plot.
        Visualizes tie point spatial positions, residual error magnitudes,
        and ISRO mandate compliance.
        """
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if not self.tie_points:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No Inlier Tie Points to Display", ha="center", va="center")
            plt.savefig(out_file, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            return out_file

        ref_x = np.array([pt["ref_x"] for pt in self.tie_points])
        ref_y = np.array([pt["ref_y"] for pt in self.tie_points])
        residuals = np.array([pt["residual_pixels"] for pt in self.tie_points])
        dx = np.array([pt["reprojected_ref_x"] - pt["ref_x"] for pt in self.tie_points])
        dy = np.array([pt["reprojected_ref_y"] - pt["ref_y"] for pt in self.tie_points])

        fig, (ax_scatter, ax_hist) = plt.subplots(
            1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2.2, 1]}
        )

        # 1. 2D Spatial Scatter Plot over Reference Image
        if background_image is not None:
            ax_scatter.imshow(background_image, cmap="gray", alpha=0.85)
        else:
            h, w = self.image_shape if self.image_shape != (0, 0) else (
                int(max(ref_y) + 20), int(max(ref_x) + 20)
            )
            ax_scatter.set_facecolor("#111111")
            ax_scatter.set_xlim(0, w)
            ax_scatter.set_ylim(h, 0)  # Inverted Y for image coordinates

        sc = ax_scatter.scatter(
            ref_x,
            ref_y,
            c=residuals,
            cmap="turbo",
            s=45,
            edgecolors="black",
            linewidths=0.6,
            vmin=0.0,
            vmax=max(0.60, float(np.percentile(residuals, 98))),
        )
        cbar = plt.colorbar(sc, ax=ax_scatter, pad=0.02, shrink=0.85)
        cbar.set_label("Reprojection Residual Error ||x_ref - H * x_src|| (pixels)", fontsize=10)

        # Draw scaled displacement error quivers
        ax_scatter.quiver(
            ref_x, ref_y, dx, dy,
            color="white",
            angles="xy",
            scale_units="xy",
            scale=0.2,
            width=0.004,
            alpha=0.75,
        )

        mandate_str = "PASSED (< 0.40 px)" if self.meets_isro_mandate else "FAILED (>= 0.40 px)"
        badge_color = "#2ca02c" if self.meets_isro_mandate else "#d62728"

        ax_scatter.set_title(
            f"Planetary Tie Point Residual Scatter Field\n"
            f"N={len(ref_x)} | Inlier Ratio={self.inlier_ratio_percent:.1f}% | "
            f"Spatial Uniformity H={self.spatial_uniformity_entropy:.2f}",
            fontsize=11,
            fontweight="bold",
        )
        ax_scatter.set_xlabel("Reference X (pixels)")
        ax_scatter.set_ylabel("Reference Y (pixels)")

        # 2. Residual Distribution Histogram
        n_bins = max(5, min(20, len(residuals) // 2))
        ax_hist.hist(residuals, bins=n_bins, color="#1f77b4", edgecolor="black", alpha=0.75)
        ax_hist.axvline(
            self.rmse_pixels,
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"RMSE: {self.rmse_pixels:.3f} px",
        )
        ax_hist.axvline(
            0.40,
            color="red",
            linestyle=":",
            linewidth=2,
            label="ISRO Mandate: 0.40 px",
        )
        ax_hist.axvline(
            self.ce90_pixels,
            color="magenta",
            linestyle="-.",
            linewidth=1.5,
            label=f"CE90: {self.ce90_pixels:.3f} px",
        )
        ax_hist.set_title("Residual Error Distribution", fontsize=11, fontweight="bold")
        ax_hist.set_xlabel("Residual Magnitude (pixels)")
        ax_hist.set_ylabel("Tie Point Count")
        ax_hist.legend(loc="upper right", fontsize=9)
        ax_hist.grid(True, linestyle="--", alpha=0.5)

        # Performance summary card annotation
        summary_text = (
            f"RMSE      : {self.rmse_pixels:.4f} px\n"
            f"Mean Res  : {self.mean_residual_pixels:.4f} px\n"
            f"CE90      : {self.ce90_pixels:.4f} px\n"
            f"Entropy H : {self.spatial_uniformity_entropy:.3f}\n"
            f"Mandate   : {mandate_str}"
        )
        ax_hist.text(
            0.05, 0.95,
            summary_text,
            transform=ax_hist.transAxes,
            verticalalignment="top",
            fontsize=9,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=badge_color, alpha=0.2),
        )

        plt.tight_layout()
        plt.savefig(out_file, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return out_file


class EvaluationEngine:
    """
    Core Evaluation Engine for SIH PS 26166 lunar image correspondence.
    """

    @staticmethod
    def compute_inlier_ratio(inlier_count: int, total_matches: int) -> float:
        """
        Computes Inlier Ratio (%) = (Inlier Count / Total Matches) * 100.
        """
        if total_matches <= 0:
            return 0.0
        return float(inlier_count / total_matches) * 100.0

    @staticmethod
    def compute_projective_rmse(
        ref_pts: np.ndarray,
        src_pts: np.ndarray,
        H: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        r"""
        Computes sub-pixel registration RMSE against verified projective tie points:
            RMSE = sqrt( 1/N * sum( ||x_ref - H * x_src||^2 ) )

        Returns:
            (rmse, residuals, reprojected_ref_pts)
        """
        ref_pts = np.asarray(ref_pts, dtype=np.float64)
        src_pts = np.asarray(src_pts, dtype=np.float64)
        n = len(ref_pts)

        if n == 0 or H is None:
            return 999.0, np.array([]), np.array([])

        # Handle 2x3 affine matrix or 3x3 homography matrix
        if H.shape == (2, 3):
            src_h = np.hstack([src_pts, np.ones((n, 1), dtype=np.float64)])
            reproj_ref = src_h @ H.T
        elif H.shape == (3, 3):
            src_h = np.hstack([src_pts, np.ones((n, 1), dtype=np.float64)])
            proj = (H @ src_h.T).T
            denom = proj[:, 2:3]
            denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
            reproj_ref = proj[:, :2] / denom
        else:
            raise ValueError(f"Transformation matrix must be 2x3 or 3x3, got {H.shape}")

        residuals = np.linalg.norm(ref_pts - reproj_ref, axis=1)
        rmse = float(np.sqrt(np.mean(residuals**2)))
        return rmse, residuals, reproj_ref

    @staticmethod
    def compute_spatial_entropy(
        keypoints: np.ndarray,
        image_shape: Tuple[int, int],
        grid_bins: int = 8,
    ) -> float:
        r"""
        Computes 2D Spatial Distribution Uniformity:
            H_spatial = - sum_{k=1}^K p_k * log2(p_k) / log2(K)
        across an image grid of K = grid_bins^2 cells.
        Score of 1.0 indicates maximum spatial uniformity (absence of crater rim clumping).
        """
        keypoints = np.asarray(keypoints)
        if len(keypoints) == 0:
            return 0.0

        h, w = image_shape[:2]
        if h <= 0 or w <= 0:
            h = max(1.0, float(np.max(keypoints[:, 1]) + 1.0))
            w = max(1.0, float(np.max(keypoints[:, 0]) + 1.0))

        hist, _, _ = np.histogram2d(
            keypoints[:, 1], keypoints[:, 0],
            bins=grid_bins,
            range=[[0, h], [0, w]],
        )
        counts = hist.flatten()
        total = np.sum(counts)
        if total == 0:
            return 0.0

        probs = counts[counts > 0] / total
        shannon = -np.sum(probs * np.log2(probs))
        max_h = np.log2(grid_bins * grid_bins)
        return float(np.clip(shannon / max_h, 0.0, 1.0))

    @classmethod
    def evaluate(
        cls,
        total_matches: int,
        inliers: List[KeypointMatch],
        image_shape: Tuple[int, int],
        homography: Optional[np.ndarray] = None,
        processing_time_ms: float = 0.0,
    ) -> RegistrationMetrics:
        """
        Computes standard RegistrationMetrics with backwards compatibility.
        """
        report = cls.generate_report(
            total_matches=total_matches,
            inliers=inliers,
            image_shape=image_shape,
            homography=homography,
            processing_time_ms=processing_time_ms,
        )
        return report.metrics

    @classmethod
    def generate_report(
        cls,
        total_matches: int,
        inliers: List[KeypointMatch],
        image_shape: Tuple[int, int],
        homography: Optional[np.ndarray] = None,
        processing_time_ms: float = 0.0,
        ground_truth_control_points: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> RegistrationEvaluationReport:
        """
        Full diagnostic evaluator returning a comprehensive RegistrationEvaluationReport.
        """
        inlier_count = len(inliers)
        inlier_ratio_pct = cls.compute_inlier_ratio(inlier_count, total_matches)
        control_point_rmse: Optional[float] = None

        if inlier_count >= 4:
            ref_pts = np.array([m.ref_xy for m in inliers], dtype=np.float64)
            src_pts = np.array([m.target_xy for m in inliers], dtype=np.float64)

            # Compute projective RMSE
            if homography is not None:
                rmse, residuals, reproj_ref = cls.compute_projective_rmse(ref_pts, src_pts, homography)
            else:
                res_list = [m.residual_error for m in inliers if m.residual_error is not None]
                if res_list:
                    residuals = np.array(res_list, dtype=np.float64)
                    rmse = float(np.sqrt(np.mean(residuals**2)))
                    reproj_ref = ref_pts
                else:
                    rmse, residuals, reproj_ref = 999.0, np.array([]), ref_pts

            if len(residuals) > 0:
                mean_res = float(np.mean(residuals))
                median_res = float(np.median(residuals))
                max_res = float(np.max(residuals))
                std_res = float(np.std(residuals))
                ce90 = float(np.percentile(residuals, 90))
            else:
                mean_res, median_res, max_res, std_res, ce90 = 999.0, 999.0, 999.0, 0.0, 999.0

            entropy = cls.compute_spatial_entropy(ref_pts, image_shape, grid_bins=8)

            # Package individual tie points
            tie_points: List[Dict[str, Any]] = []
            for i, m in enumerate(inliers):
                res_val = float(residuals[i]) if i < len(residuals) else 0.0
                rx_p = float(reproj_ref[i, 0]) if i < len(reproj_ref) else float(m.ref_xy[0])
                ry_p = float(reproj_ref[i, 1]) if i < len(reproj_ref) else float(m.ref_xy[1])
                tie_points.append({
                    "id": i,
                    "ref_x": float(m.ref_xy[0]),
                    "ref_y": float(m.ref_xy[1]),
                    "src_x": float(m.target_xy[0]),
                    "src_y": float(m.target_xy[1]),
                    "reprojected_ref_x": rx_p,
                    "reprojected_ref_y": ry_p,
                    "residual_pixels": res_val,
                    "confidence": float(m.confidence),
                    "subpixel_refined": m.subpixel_refined,
                })

            h_list = homography.tolist() if homography is not None else None
            meets_mandate = bool(rmse < 0.40 and inlier_count >= 4)

            # Evaluate against ground-truth control points if provided
            if ground_truth_control_points is not None and homography is not None:
                gt_ref, gt_src = ground_truth_control_points
                if len(gt_ref) > 0:
                    cp_rmse, _, _ = cls.compute_projective_rmse(gt_ref, gt_src, homography)
                    control_point_rmse = float(cp_rmse)
        else:
            rmse, mean_res, median_res, max_res, std_res, ce90, entropy = (
                999.0, 999.0, 999.0, 999.0, 0.0, 999.0, 0.0
            )
            tie_points = []
            h_list = None
            meets_mandate = False

        return RegistrationEvaluationReport(
            total_matches=total_matches,
            inlier_count=inlier_count,
            inlier_ratio_percent=inlier_ratio_pct,
            rmse_pixels=rmse,
            spatial_uniformity_entropy=entropy,
            mean_residual_pixels=mean_res,
            median_residual_pixels=median_res,
            max_residual_pixels=max_res,
            std_residual_pixels=std_res,
            ce90_pixels=ce90,
            meets_isro_mandate=meets_mandate,
            control_point_rmse_pixels=control_point_rmse,
            processing_time_ms=processing_time_ms,
            homography_matrix=h_list,
            tie_points=tie_points,
            image_shape=image_shape,
        )
