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
