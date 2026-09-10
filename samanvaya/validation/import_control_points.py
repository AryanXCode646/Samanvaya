from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

VALID_METHODS = {"manual", "independent_reference", "photogrammetric"}


def _validate_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for idx, point in enumerate(points):
        if not isinstance(point, dict):
            raise ValueError(f"Control point at index {idx} must be an object.")
        required = {"source_x", "source_y", "reference_x", "reference_y"}
        missing = sorted(required - set(point))
        if missing:
            raise ValueError(f"Control point at index {idx} is missing required fields: {missing}")
        uncertainty = point.get("uncertainty_px", 0.0)
        validated.append({
            "source_x": float(point["source_x"]),
            "source_y": float(point["source_y"]),
            "reference_x": float(point["reference_x"]),
            "reference_y": float(point["reference_y"]),
            "uncertainty_px": float(uncertainty),
        })
    if len(validated) < 1:
        raise ValueError("At least one control point is required.")
    return validated


def import_control_points(
    pair_id: str,
    file_path: str | Path,
    *,
    manifest_path: str | Path = "data/real/manifest.json",
) -> Path:
    source = Path(file_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Control-point file not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Control-point payload must be a JSON object.")

    method = str(payload.get("method", "")).lower()
    if method not in VALID_METHODS:
        raise ValueError(f"Unsupported control-point method '{method}'. Allowed: {sorted(VALID_METHODS)}")

    pair_value = payload.get("pair_id")
    if pair_value is not None and str(pair_value) != str(pair_id):
        raise ValueError(f"Control-point file pair_id '{pair_value}' does not match requested pair_id '{pair_id}'.")

    points = payload.get("points", [])
    if not isinstance(points, list):
        raise ValueError("Control-point payload field 'points' must be a list.")
    validated_points = _validate_points(points)

    saved = {
        "pair_id": pair_id,
        "method": method,
        "source": payload.get("source", "unknown"),
        "points": validated_points,
    }
    output_dir = Path(manifest_path).expanduser().resolve().parent / "control_points"
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{pair_id}.json"
    target.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    return target


def _main() -> None:
    parser = argparse.ArgumentParser(description="Import independent control points for a real pair.")
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--file", required=True, help="Path to a JSON control-point file")
    parser.add_argument("--manifest", default="data/real/manifest.json", help="Manifest path")
    args = parser.parse_args()
    result = import_control_points(args.pair_id, args.file, manifest_path=args.manifest)
    print(f"Imported control points for {args.pair_id}: {result}")


if __name__ == "__main__":
    _main()
