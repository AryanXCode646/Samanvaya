"""Scientific-validation helpers based on evidence manifests.

This module keeps the repository honest by generating a machine-readable validation
summary from the evidence package instead of leaving claims in README prose alone.
The project remains intentionally conservative here: unless a real-data experiment is
present in the manifest, the validation matrix reports the pair as unvalidated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_PAIRS: dict[str, dict[str, str]] = {
    "OHRC ↔ TMC-2": {
        "ingestion": "REAL DATA TESTED",
        "registration": "NOT RUN",
        "ground_truth": "NOT AVAILABLE",
        "status": "NOT VALIDATED",
    },
    "OHRC ↔ IIRS": {
        "ingestion": "REAL DATA TESTED",
        "registration": "NOT RUN",
        "ground_truth": "NOT AVAILABLE",
        "status": "NOT VALIDATED",
    },
    "TMC-2 ↔ IIRS": {
        "ingestion": "REAL DATA TESTED",
        "registration": "NOT RUN",
        "ground_truth": "NOT AVAILABLE",
        "status": "NOT VALIDATED",
    },
    "Chandrayaan-2 ↔ LRO NAC": {
        "ingestion": "NOT AVAILABLE",
        "registration": "NOT RUN",
        "ground_truth": "NOT AVAILABLE",
        "status": "NOT VALIDATED",
    },
    "Chandrayaan-2 ↔ SELENE": {
        "ingestion": "NOT AVAILABLE",
        "registration": "NOT RUN",
        "ground_truth": "NOT AVAILABLE",
        "status": "NOT VALIDATED",
    },
}


def _load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_validation_matrix(path: str | Path) -> dict[str, dict[str, str]]:
    """Return a validation matrix keyed by pair name.

    The perspective is intentionally conservative: the matrix contains the required
    pair taxonomy and only upgrades a pair when the evidence explicitly supports a
    claim. Missing mission archives are recorded as NOT AVAILABLE and no numeric
    metrics are fabricated.
    """

    manifest = _load_manifest(path)
    products = manifest.get("products", [])
    product_ids = {product.get("product_id") for product in products if product.get("product_id")}
    capabilities = manifest.get("capabilities", {})

    matrix: dict[str, dict[str, str]] = {}
    for pair_name, row in REQUIRED_PAIRS.items():
        pair_row = dict(row)
        pair_row["N"] = "N/A"
        pair_row["success_rate"] = "N/A"
        pair_row["median_gt_rmse"] = "N/A"
        pair_row["median_reproj_rmse"] = "N/A"

        if product_ids:
            if "OHRC" in {product.get("instrument") for product in products}:
                pair_row["ingestion"] = "REAL DATA TESTED"
        if capabilities.get("registration_executed"):
            pair_row["registration"] = "REAL DATA EXECUTED"
        if capabilities.get("ground_truth_available"):
            pair_row["ground_truth"] = "AVAILABLE"
        if pair_name == "OHRC ↔ TMC-2" and not capabilities.get("registration_executed"):
            pair_row["registration"] = "NOT RUN"
        if pair_name == "OHRC ↔ TMC-2" and not capabilities.get("ground_truth_available"):
            pair_row["ground_truth"] = "NOT AVAILABLE"

        if pair_row["registration"] == "REAL DATA EXECUTED" and pair_row["ground_truth"] == "AVAILABLE":
            pair_row["status"] = "EVIDENCE-BASED"
        else:
            pair_row["status"] = "NOT VALIDATED"

        matrix[pair_name] = pair_row

    return matrix


def summarize_validation_evidence(path: str | Path) -> dict[str, Any]:
    """Materialize a conservative scientific-validation summary for the repo."""

    manifest = _load_manifest(path)
    matrix = build_validation_matrix(path)
    return {
        "validation_scope": manifest.get("validation_scope"),
        "real_image_validation_status": manifest.get("real_image_validation_status"),
        "merge_readiness_scope": manifest.get("merge_readiness_scope"),
        "capabilities": manifest.get("capabilities", {}),
        "scientific_claims": manifest.get("scientific_claims", []),
        "validation_matrix": matrix,
    }
