"""Manifest-driven real benchmark execution with explicit DATA_REQUIRED states."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from lunar_core.data_io.mission_catalog import inspect_product
from samanvaya.provenance import git_commit_sha
from samanvaya.validation.real_registration import RealRegistrationResult, register_products
from samanvaya.validation.claim_gate import evaluate_claims
from samanvaya.validation.checkpoints import load_checkpoints


def _status_value(product: Any) -> str:
    return str(getattr(getattr(product, "status", None), "value", getattr(product, "status", "")))


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _failure(pair_id: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "status": "DATA_REQUIRED",
        "validation_status": "DATA_REQUIRED",
        "reason": reason,
        **extra,
    }


def _evaluate_checkpoints(checkpoint_path: Path, transform: Any, source_shape: tuple[int, int], reference_shape: tuple[int, int]) -> dict[str, Any]:
    points = load_checkpoints(checkpoint_path, source_shape=source_shape, reference_shape=reference_shape)
    if len(points) < 4:
        return {"status": "CHECKPOINT_VALIDATION_FAILED", "checkpoint_count": len(points), "reason": "At least four checkpoints are required."}
    matrix = np.asarray(transform, dtype=float)
    source = np.asarray([[point["source_x"], point["source_y"]] for point in points], dtype=float)
    reference = np.asarray([[point["reference_x"], point["reference_y"]] for point in points], dtype=float)
    homogeneous = np.column_stack([source, np.ones(len(source))])
    projected = homogeneous @ matrix.T
    projected = projected[:, :2] / projected[:, 2:3]
    errors = np.linalg.norm(projected - reference, axis=1)
    return {
        "status": "READY",
        "checkpoint_count": len(points),
        "valid_checkpoint_count": int(np.isfinite(errors).sum()),
        "rmse_pixels": float(np.sqrt(np.mean(errors ** 2))),
        "median_error_pixels": float(np.median(errors)),
        "p95_error_pixels": float(np.percentile(errors, 95)),
        "max_error_pixels": float(np.max(errors)),
        "mean_error_pixels": float(np.mean(errors)),
    }


def run_real_benchmark(manifest_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not manifest_file.is_file():
        result = {"status": "DATA_REQUIRED", "reason": f"Manifest not found: {manifest_file}", "results": []}
        (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    pairs = payload.get("pairs", [])
    if not pairs:
        result = {"status": "DATA_REQUIRED", "reason": "Manifest contains no executable benchmark pairs.", "results": []}
        (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (output / "summary.md").write_text("# Real benchmark\n\nDATA_REQUIRED: manifest contains no pairs.\n", encoding="utf-8")
        return result

    rows: list[dict[str, Any]] = []
    for pair in pairs:
        pair_id = str(pair.get("pair_id", "UNNAMED_PAIR"))
        source_meta = pair.get("source_product") or {}
        reference_meta = pair.get("reference_product") or {}
        source_path = Path(source_meta.get("image_path", "")).expanduser().resolve()
        reference_path = Path(reference_meta.get("image_path", "")).expanduser().resolve()
        missing = [str(path) for path in (source_path, reference_path) if not path.is_file()]
        if missing:
            row = _failure(pair_id, "Required source/reference raster is missing.", missing_products=missing)
            (output / f"{pair_id}.failure.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            rows.append(row)
            continue
        try:
            source_product = inspect_product(source_path, root_dir=source_path.parent)
            reference_product = inspect_product(reference_path, root_dir=reference_path.parent)
            if _status_value(source_product) not in {"validated", "partial"}:
                raise ValueError(f"INVALID_SOURCE_PRODUCT: {source_product.validation_message}")
            if _status_value(reference_product) not in {"validated", "partial"}:
                raise ValueError(f"INVALID_REFERENCE_PRODUCT: {reference_product.validation_message}")
            result = register_products(source_product, reference_product, output / pair_id)
            row = result.to_dict()
            row["pair_id"] = pair_id
            row["source_sha256"] = _hash_file(source_path)
            row["reference_sha256"] = _hash_file(reference_path)
            row["repository_commit"] = git_commit_sha()
            ground_truth = pair.get("ground_truth")
            if ground_truth and ground_truth.get("path"):
                checkpoint_path = Path(ground_truth["path"]).expanduser().resolve()
                if checkpoint_path.is_file() and row.get("transform_parameters") is not None:
                    row["held_out_metrics"] = _evaluate_checkpoints(
                        checkpoint_path,
                        row["transform_parameters"],
                        (source_product.height or 0, source_product.width or 0),
                        (reference_product.height or 0, reference_product.width or 0),
                    )
                    (output / pair_id / "held_out_metrics.json").write_text(json.dumps(row["held_out_metrics"], indent=2), encoding="utf-8")
                else:
                    row["held_out_metrics"] = {"status": "DATA_REQUIRED", "reason": "Checkpoint path or transform is unavailable."}
            else:
                row["held_out_metrics"] = {"status": "DATA_REQUIRED", "reason": "No independent checkpoint file supplied."}
            row["claims"] = evaluate_claims(row)
            if row.get("status") != "SUCCESS":
                failure = {
                    "stage": "registration",
                    "status": row.get("status"),
                    "reason": row.get("failure_reason") or "Registration did not produce a scientifically acceptable result.",
                    "diagnostics": row,
                    "last_successful_stage": "product_validation",
                    "provenance": {"repository_commit": git_commit_sha(), "pair_id": pair_id},
                }
                (output / f"{pair_id}.failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        except (OSError, ValueError, KeyError) as exc:
            row = _failure(pair_id, str(exc), last_successful_stage="product_validation")
            (output / f"{pair_id}.failure.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)

    results = {"status": "COMPLETE", "repository_commit": git_commit_sha(), "results": rows}
    (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    fields = ["pair_id", "status", "validation_status", "source_id", "reference_id", "raw_match_count", "inlier_count", "inlier_ratio", "coverage_fraction", "failure_reason", "reason"]
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
    summary_lines = ["# Real benchmark", "", f"Repository commit: `{git_commit_sha()}`", ""]
    for row in rows:
        summary_lines.append(f"- `{row.get('pair_id')}`: **{row.get('status')}** — {row.get('reason') or row.get('failure_reason') or row.get('validation_status')}")
    (output / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return results
