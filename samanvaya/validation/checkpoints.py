"""Strict independent checkpoint parsing for real validation."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

REQUIRED = {"point_id", "source_x", "source_y", "reference_x", "reference_y", "provenance", "quality"}


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
    import numpy as np

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

