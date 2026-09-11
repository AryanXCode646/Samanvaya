"""Strict independent checkpoint parsing and held-out validation for planetary registration."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import numpy as np

REQUIRED = {"point_id", "source_x", "source_y", "reference_x", "reference_y", "provenance", "quality"}


@dataclass
class IndependentValidationReport:
    """
    Evaluation report strictly separating transformation-estimation control points
    from independent, held-out validation checkpoints.
    """
    control_points_count: int
    independent_checkpoints_count: int
    consensus_rmse_pixels: float
    checkpoint_rmse_pixels: float
    checkpoint_mae_pixels: float
    checkpoint_median_pixels: float
    checkpoint_p95_pixels: float
    checkpoint_max_pixels: float
    inlier_ratio: float
    status: str = "INDEPENDENTLY_VALIDATED"
    checkpoint_residuals: List[float] = field(default_factory=list)
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_type: str = "HOMOGRAPHY"
    meets_isro_mandate: bool = False

    def __post_init__(self) -> None:
        self.meets_isro_mandate = bool(
            self.checkpoint_rmse_pixels < 0.40 and self.independent_checkpoints_count >= 4
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": {
                "validation_paradigm": "STRICT_INDEPENDENT_CHECKPOINTS",
                "timestamp_utc": self.timestamp_utc,
                "model_type": self.model_type,
                "status": self.status,
            },
            "summary": {
                "control_points_count": self.control_points_count,
                "independent_checkpoints_count": self.independent_checkpoints_count,
                "consensus_rmse_pixels": round(self.consensus_rmse_pixels, 4),
                "checkpoint_rmse_pixels": round(self.checkpoint_rmse_pixels, 4),
                "checkpoint_mae_pixels": round(self.checkpoint_mae_pixels, 4),
                "checkpoint_median_pixels": round(self.checkpoint_median_pixels, 4),
                "checkpoint_p95_pixels": round(self.checkpoint_p95_pixels, 4),
                "checkpoint_max_pixels": round(self.checkpoint_max_pixels, 4),
                "inlier_ratio": round(self.inlier_ratio, 4),
                "meets_isro_mandate": self.meets_isro_mandate,
                "mandate_threshold_px": 0.40,
            },
            "checkpoint_residuals": [round(float(r), 4) for r in self.checkpoint_residuals],
        }

    def export_json(self, path: Union[str, Path], indent: int = 2) -> str:
        out_p = Path(path).expanduser().resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), indent=indent)
        out_p.write_text(text, encoding="utf-8")
        return text

    def export_csv(self, path: Union[str, Path]) -> str:
        out_p = Path(path).expanduser().resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# SAMANVAYA INDEPENDENT CHECKPOINT EVALUATION",
            "# Metric,Value,Unit",
            f"control_points_count,{self.control_points_count},count",
            f"independent_checkpoints_count,{self.independent_checkpoints_count},count",
            f"consensus_rmse_pixels,{self.consensus_rmse_pixels:.4f},pixels",
            f"checkpoint_rmse_pixels,{self.checkpoint_rmse_pixels:.4f},pixels",
            f"checkpoint_mae_pixels,{self.checkpoint_mae_pixels:.4f},pixels",
            f"checkpoint_median_pixels,{self.checkpoint_median_pixels:.4f},pixels",
            f"checkpoint_p95_pixels,{self.checkpoint_p95_pixels:.4f},pixels",
            f"checkpoint_max_pixels,{self.checkpoint_max_pixels:.4f},pixels",
            f"inlier_ratio,{self.inlier_ratio:.4f},ratio",
            f"meets_isro_mandate,{self.meets_isro_mandate},boolean",
        ]
        text = "\n".join(lines) + "\n"
        out_p.write_text(text, encoding="utf-8")
        return text


def partition_control_and_checkpoints(
    matches: list[Any],
    control_fraction: float = 0.70,
    seed: int = 42,
) -> Tuple[list[Any], list[Any]]:
    """
    Partitions correspondence points into estimation control points (e.g. 70%)
    and strictly held-out independent validation checkpoints (e.g. 30%).
    Validation checkpoints NEVER enter or influence the model estimation step.
    """
    if len(matches) < 6:
        return list(matches), []

    rng = np.random.RandomState(seed)
    indices = np.arange(len(matches))
    rng.shuffle(indices)

    num_control = max(4, int(round(len(matches) * control_fraction)))
    if len(matches) - num_control < 2:
        num_control = len(matches) - 2

    control_idx = indices[:num_control]
    checkpoint_idx = indices[num_control:]

    control_points = [matches[i] for i in sorted(control_idx)]
    checkpoint_points = [matches[i] for i in sorted(checkpoint_idx)]
    return control_points, checkpoint_points


def evaluate_independent_checkpoints(
    transform_matrix: np.ndarray,
    control_points: list[Any],
    checkpoints: list[Any],
    inlier_ratio: float = 1.0,
    model_type: str = "HOMOGRAPHY",
) -> IndependentValidationReport:
    """
    Evaluates a fitted transformation against strictly held-out checkpoints.
    Computes consensus error on control points and independent accuracy on checkpoints.
    """
    matrix = np.asarray(transform_matrix, dtype=float)

    def _eval_pts(pts: list[Any]) -> Tuple[float, np.ndarray]:
        if not pts:
            return 999.0, np.array([])
        src = np.asarray([[float(m.target_xy[0]), float(m.target_xy[1])] for m in pts], dtype=float)
        ref = np.asarray([[float(m.ref_xy[0]), float(m.ref_xy[1])] for m in pts], dtype=float)
        homo = np.column_stack([src, np.ones(len(src))])
        proj = homo @ matrix.T
        denom = np.where(np.abs(proj[:, 2:3]) < 1e-12, 1e-12, proj[:, 2:3])
        proj_xy = proj[:, :2] / denom
        res = np.linalg.norm(proj_xy - ref, axis=1)
        rmse = float(np.sqrt(np.mean(res ** 2)))
        return rmse, res

    consensus_rmse, _ = _eval_pts(control_points)
    chk_rmse, chk_res = _eval_pts(checkpoints)

    if len(chk_res) > 0:
        chk_mae = float(np.mean(chk_res))
        chk_med = float(np.median(chk_res))
        chk_p95 = float(np.percentile(chk_res, 95))
        chk_max = float(np.max(chk_res))
        status = "INDEPENDENTLY_VALIDATED"
    else:
        chk_mae, chk_med, chk_p95, chk_max = 999.0, 999.0, 999.0, 999.0
        status = "NO_INDEPENDENT_CHECKPOINTS"

    return IndependentValidationReport(
        control_points_count=len(control_points),
        independent_checkpoints_count=len(checkpoints),
        consensus_rmse_pixels=consensus_rmse,
        checkpoint_rmse_pixels=chk_rmse,
        checkpoint_mae_pixels=chk_mae,
        checkpoint_median_pixels=chk_med,
        checkpoint_p95_pixels=chk_p95,
        checkpoint_max_pixels=chk_max,
        inlier_ratio=inlier_ratio,
        status=status,
        checkpoint_residuals=chk_res.tolist(),
        model_type=model_type,
    )


def load_checkpoints(path: str | Path, *, source_shape: tuple[int, int] | None = None, reference_shape: tuple[int, int] | None = None) -> list[dict[str, Any]]:
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {file_path}")
    if file_path.suffix.lower() == ".csv":
        with file_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    else:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        rows = payload.get("points", payload) if isinstance(payload, (dict, list)) else []
    if not isinstance(rows, list) or not rows:
        raise ValueError("Checkpoint file contains no points")
    validated: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not REQUIRED.issubset(row):
            raise ValueError(f"Checkpoint {index} requires {sorted(REQUIRED)}")
        point_id = str(row["point_id"])
        if point_id in ids:
            raise ValueError(f"Duplicate checkpoint ID: {point_id}")
        ids.add(point_id)
        try:
            values = {key: float(row[key]) for key in ("source_x", "source_y", "reference_x", "reference_y")}
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid coordinate at checkpoint {point_id}") from exc
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"Non-finite coordinate at checkpoint {point_id}")
        if not str(row["provenance"]).strip():
            raise ValueError(f"Missing provenance at checkpoint {point_id}")
        if str(row["quality"]).upper() not in {"A", "B", "C"}:
            raise ValueError(f"Invalid checkpoint quality at checkpoint {point_id}")
        if source_shape and not (0 <= values["source_x"] < source_shape[1] and 0 <= values["source_y"] < source_shape[0]):
            raise ValueError(f"Source coordinate outside bounds at checkpoint {point_id}")
        if reference_shape and not (0 <= values["reference_x"] < reference_shape[1] and 0 <= values["reference_y"] < reference_shape[0]):
            raise ValueError(f"Reference coordinate outside bounds at checkpoint {point_id}")
        validated.append({"point_id": point_id, **values, "provenance": str(row["provenance"]), "quality": str(row["quality"]).upper(), "notes": row.get("notes", "")})
    return validated


def evaluate_checkpoints(transform: Any, points: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate transformation matrix against independent ground-truth checkpoints."""
    if len(points) < 4:
        return {"status": "CHECKPOINT_VALIDATION_FAILED", "checkpoint_count": len(points), "reason": "At least four checkpoints are required."}
    matrix = np.asarray(transform, dtype=float)
    source = np.asarray([[point["source_x"], point["source_y"]] for point in points], dtype=float)
    reference = np.asarray([[point["reference_x"], point["reference_y"]] for point in points], dtype=float)
    homogeneous = np.column_stack([source, np.ones(len(source))])
    projected = homogeneous @ matrix.T
    projected = projected[:, :2] / projected[:, 2:3]
    errors = np.linalg.norm(projected - reference, axis=1)

    integer_projected = np.round(projected)
    int_errors = np.linalg.norm(integer_projected - reference, axis=1)

    subpixel_rmse = float(np.sqrt(np.mean(errors ** 2)))
    integer_rmse = float(np.sqrt(np.mean(int_errors ** 2)))
    subpixel_median = float(np.median(errors))
    integer_median = float(np.median(int_errors))
    subpixel_p95 = float(np.percentile(errors, 95))
    integer_p95 = float(np.percentile(int_errors, 95))
    improvement = float(((integer_rmse - subpixel_rmse) / max(integer_rmse, 1e-6)) * 100.0) if integer_rmse > 0 else 0.0

    return {
        "status": "READY",
        "checkpoint_count": len(points),
        "valid_checkpoint_count": int(np.isfinite(errors).sum()),
        "rmse_pixels": subpixel_rmse,
        "subpixel_rmse_pixels": subpixel_rmse,
        "integer_rmse_pixels": integer_rmse,
        "median_error_pixels": subpixel_median,
        "subpixel_median_pixels": subpixel_median,
        "integer_median_pixels": integer_median,
        "p95_error_pixels": subpixel_p95,
        "subpixel_p95_pixels": subpixel_p95,
        "integer_p95_pixels": integer_p95,
        "improvement_percentage": improvement,
        "max_error_pixels": float(np.max(errors)),
        "mean_error_pixels": float(np.mean(errors)),
    }
