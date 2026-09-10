"""Utilities for validating and summarizing mission dataset manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED_FIELDS = {
    "id",
    "mission",
    "spacecraft",
    "instrument",
    "product_id",
    "product_level",
    "acquisition_utc",
    "latitude",
    "longitude",
    "resolution_m_per_px",
    "image_dimensions",
    "geometry_available",
    "calibration_status",
    "source",
    "source_url",
    "local_path",
    "checksum_sha256",
    "license",
    "processing_status",
    "registration_suitability",
    "validation_status",
    "ground_truth_source",
    "notes",
}

VALID_STATUS_VALUES = {"pending", "available", "validated", "partial", "failed", "not_tested"}


def load_dataset_manifest(path: str | Path) -> dict[str, Any]:
    """Load a YAML dataset manifest."""
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict) or "datasets" not in data:
        raise ValueError(f"Manifest at {manifest_path} must contain a top-level 'datasets' list.")
    datasets = data["datasets"]
    if not isinstance(datasets, list):
        raise TypeError("'datasets' entry must be a list of dataset objects.")
    for index, entry in enumerate(datasets):
        validate_dataset_entry(entry, index=index)
    return data


def validate_dataset_entry(entry: dict[str, Any], index: int | None = None) -> dict[str, Any]:
    """Validate a manifest entry and return it unchanged."""
    if not isinstance(entry, dict):
        raise TypeError(f"Dataset entry at index {index} must be a dictionary.")
    missing = sorted(field for field in REQUIRED_FIELDS if field not in entry)
    if missing:
        raise ValueError(f"Dataset entry at index {index} is missing required fields: {missing}")

    validation_status = str(entry.get("validation_status", "")).lower()
    if validation_status not in VALID_STATUS_VALUES:
        raise ValueError(
            f"Dataset entry at index {index} has invalid validation_status '{validation_status}'. "
            f"Allowed values: {sorted(VALID_STATUS_VALUES)}"
        )

    return entry


def summarize_dataset_status(path: str | Path) -> dict[str, Any]:
    """Summarize mission availability and dataset readiness."""
    manifest = load_dataset_manifest(path)
    datasets = manifest["datasets"]
    summary = {
        "total_datasets": len(datasets),
        "by_mission": {},
        "by_status": {},
        "pending_real_data": [],
    }

    for dataset in datasets:
        mission = str(dataset.get("mission", "unknown"))
        status = str(dataset.get("validation_status", "pending")).lower()
        summary["by_mission"].setdefault(mission, 0)
        summary["by_mission"][mission] += 1
        summary["by_status"].setdefault(status, 0)
        summary["by_status"][status] += 1
        if status in {"pending", "not_tested"}:
            summary["pending_real_data"].append(dataset.get("id"))

    return summary
