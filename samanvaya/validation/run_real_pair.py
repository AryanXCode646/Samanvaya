from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from lunar_core.pipeline import LunarCorePipeline
from lunar_core.models import SunAngles, TransformationType
from samanvaya.data_io.import_real_pair import DEFAULT_ARTIFACT_ROOT, DEFAULT_MANIFEST_PATH
from samanvaya.provenance import config_hash, git_commit_sha


def _make_mock_source_reference(source: np.ndarray, reference: np.ndarray):
    return source.astype(np.float32), reference.astype(np.float32)


def _pair_entry(manifest_path: Path, pair_id: str) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    pairs = payload.get("pairs", [])
    for entry in pairs:
        if str(entry.get("pair_id")) == str(pair_id):
            return entry
    raise KeyError(f"Pair ID {pair_id!r} was not found in the manifest.")


def _read_image_array(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def run_real_pair(pair_id: str, *, manifest_path: str | Path = DEFAULT_MANIFEST_PATH, artifact_root: str | Path | None = None) -> dict[str, Any]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")

    pair = _pair_entry(manifest_file, pair_id)
    if artifact_root is None:
        artifact_root = manifest_file.parent / "artifacts" / "real_validation"
    artifact_root = Path(artifact_root).expanduser().resolve()

    source_meta = pair["source_product"]
    ref_meta = pair["reference_product"]
    source_path = Path(source_meta["image_path"]).expanduser().resolve()
    ref_path = Path(ref_meta["image_path"]).expanduser().resolve()

    missing = [str(path) for path in (source_path, ref_path) if not path.is_file()]
    if missing:
        return {
            "pair_id": pair_id,
            "status": "BLOCKED_BY_MISSING_DATA",
            "ground_truth_status": "NOT_RUN",
            "reason": "Registration was not executed because the manifest references missing raster products.",
            "missing_products": missing,
            "artifact_dir": str(Path(artifact_root).expanduser().resolve() / pair_id),
            "metrics": {"rmse_px": None, "p95_px": None, "inlier_count": None},
        }

    source_data = _read_image_array(source_path)
    ref_data = _read_image_array(ref_path)

    pipeline = LunarCorePipeline(transformation_type=TransformationType.HOMOGRAPHY)
    result = pipeline.register(
        ref_data,
        source_data,
        ref_sun=SunAngles(azimuth_deg=90.0, elevation_deg=30.0),
        target_sun=SunAngles(azimuth_deg=120.0, elevation_deg=35.0),
        ref_gsd=float(source_meta.get("gsd_m") or 0.25),
        target_gsd=float(ref_meta.get("gsd_m") or 0.50),
    )

    artifact_dir = Path(artifact_root).expanduser().resolve() / pair_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    import cv2

    cv2.imwrite(str(artifact_dir / "source.png"), np.clip(source_data, 0, 255).astype(np.uint8))
    cv2.imwrite(str(artifact_dir / "reference.png"), np.clip(ref_data, 0, 255).astype(np.uint8))
    if result.warped_target is not None:
        cv2.imwrite(str(artifact_dir / "registered_source.png"), np.clip(result.warped_target, 0, 255).astype(np.uint8))

    metrics_payload = {
        "pair_id": pair_id,
        "total_matches": len(result.matches),
        "inlier_count": len(result.inliers),
        "inlier_ratio": float(getattr(result.metrics, "inlier_ratio", 0.0)),
        "estimated_transform": None if result.transform_matrix is None else result.transform_matrix.tolist(),
        "runtime_ms": float(getattr(result.metrics, "processing_time_ms", 0.0)),
        "memory_mb": 0.0,
        "ground_truth_status": "PENDING_INDEPENDENT_GROUND_TRUTH",
        "rmse_px": None,
        "p95_px": None,
        "absolute_accuracy": "PENDING_INDEPENDENT_CHECKPOINTS",
        "source_product_id": source_meta.get("product_id"),
        "reference_product_id": ref_meta.get("product_id"),
        "repository_commit": git_commit_sha(),
        "config_hash": config_hash({
            "transformation": "homography",
            "subpixel_refinement": True,
            "source_gsd": source_meta.get("gsd_m"),
            "reference_gsd": ref_meta.get("gsd_m"),
        }),
        "validation_scope": "REAL_REGISTRATION_PENDING_INDEPENDENT_VALIDATION",
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    (artifact_dir / "metadata.json").write_text(json.dumps({"pair_id": pair_id, "source": source_meta, "reference": ref_meta}, indent=2), encoding="utf-8")
    (artifact_dir / "run.log").write_text(
        f"Executed registration on supplied raster products for {pair_id}.\n"
        "Ground-truth accuracy was not inferred from fitting-point reprojection.\n",
        encoding="utf-8",
    )

    output = {
        "pair_id": pair_id,
        "status": "registration_executed",
        "ground_truth_status": "PENDING_INDEPENDENT_GROUND_TRUTH",
        "match_count": len(result.matches),
        "inlier_count": len(result.inliers),
        "inlier_ratio": float(getattr(result.metrics, "inlier_ratio", 0.0)),
        "transform": None if result.transform_matrix is None else result.transform_matrix.tolist(),
        "artifact_dir": str(artifact_dir),
        "metrics": metrics_payload,
    }
    return output


def validate_real_pair(pair_id: str, *, manifest_path: str | Path = DEFAULT_MANIFEST_PATH, artifact_root: str | Path | None = None) -> dict[str, Any]:
    """Validate a real imported OHRC ↔ LROC pair using conservative evidence rules.

    Scientific claims are only elevated to PASS when independent ground-truth control points
    exist and the measured RMSE is within the ISRO mandate. In the absence of such evidence,
    the function reports NOT_VALIDATED or FAIL with an explicit technical reason.
    """
    manifest_file = Path(manifest_path).expanduser().resolve()
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")

    pair = _pair_entry(manifest_file, pair_id)
    control_points_path = manifest_file.parent / "control_points" / f"{pair_id}.json"
    if not control_points_path.exists():
        status = {
            "pair_id": pair_id,
            "status": "NOT_VALIDATED",
            "ground_truth_status": "PENDING_INDEPENDENT_GROUND_TRUTH",
            "reason": "No independent control point file was supplied for this real pair; only reprojection consensus was computed.",
            "metrics": {
                "rmse_px": None,
                "p95_px": None,
                "inlier_count": 0,
            },
            "artifact_dir": pair.get("artifacts_dir"),
        }
        return status

    from samanvaya.validation.evaluate_real_pair import evaluate_real_pair

    evaluated = evaluate_real_pair(pair_id, manifest_path=manifest_file)
    evaluated["artifact_dir"] = pair.get("artifacts_dir")
    if evaluated.get("status") == "READY":
        rmse = evaluated.get("rmse_px")
        evaluated["status"] = "PASS" if rmse is not None and rmse < 0.40 else "FAIL"
        evaluated["reason"] = (
            "Independent held-out checkpoints were evaluated against the executed transform."
            if evaluated["status"] == "PASS"
            else "Independent held-out checkpoints were evaluated, but residuals exceed the 0.40 px mandate."
        )
    return evaluated


def _main() -> None:
    parser = argparse.ArgumentParser(description="Execute a conservative validation pass for a previously imported real OHRC ↔ LROC pair.")
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH), help="Manifest path")
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT), help="Artifact root directory")
    args = parser.parse_args()
    result = validate_real_pair(args.pair_id, manifest_path=args.manifest, artifact_root=args.artifact_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
