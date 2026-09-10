from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def evaluate_real_pair(pair_id: str, *, manifest_path: str | Path = "data/real/manifest.json") -> dict[str, Any]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")

    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    pairs = payload.get("pairs", [])
    pair = next((entry for entry in pairs if str(entry.get("pair_id")) == str(pair_id)), None)
    if pair is None:
        raise KeyError(f"No pair metadata found for {pair_id}")

    control_path = Path(manifest_file.parent) / "control_points" / f"{pair_id}.json"
    if not control_path.exists():
        return {
            "pair_id": pair_id,
            "status": "PENDING_INDEPENDENT_GROUND_TRUTH",
            "ground_truth_status": "PENDING_INDEPENDENT_GROUND_TRUTH",
            "rmse_px": None,
            "mae_px": None,
            "median_px": None,
            "p95_px": None,
            "max_px": None,
            "notes": "No independent control points were supplied for this pair.",
        }

    points = json.loads(control_path.read_text(encoding="utf-8"))["points"]
    if len(points) < 1:
        raise ValueError(f"Control-point file for {pair_id} contains no valid points.")

    source = np.array([[p["source_x"], p["source_y"]] for p in points], dtype=np.float64)
    reference = np.array([[p["reference_x"], p["reference_y"]] for p in points], dtype=np.float64)
    residuals = np.linalg.norm(source - reference, axis=1)

    result = {
        "pair_id": pair_id,
        "status": "READY",
        "ground_truth_status": "INDEPENDENT_GROUND_TRUTH_AVAILABLE",
        "rmse_px": float(np.sqrt(np.mean(residuals ** 2))),
        "mae_px": float(np.mean(residuals)),
        "median_px": float(np.median(residuals)),
        "p95_px": float(np.percentile(residuals, 95)),
        "max_px": float(np.max(residuals)),
        "per_point_residual_px": [float(x) for x in residuals],
        "count": int(len(points)),
    }
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a real pair against independently supplied control points.")
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--manifest", default="data/real/manifest.json", help="Manifest path")
    args = parser.parse_args()
    result = evaluate_real_pair(args.pair_id, manifest_path=args.manifest)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
