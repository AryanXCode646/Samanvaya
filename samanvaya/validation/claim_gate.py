"""Automatic scientific claim gates.

Claims are derived only from structured evidence; missing evidence remains DATA_REQUIRED.
"""

from __future__ import annotations

from typing import Any


def evaluate_claims(result: dict[str, Any]) -> dict[str, str]:
    status = str(result.get("status", ""))
    validation = str(result.get("validation_status", ""))
    held_out = result.get("held_out_metrics") or {}
    independent_ready = held_out.get("status") == "READY" and held_out.get("rmse_pixels") is not None
    spatial = result.get("coverage_fraction")
    inlier_ratio = result.get("inlier_ratio")
    reopened = bool(result.get("registered_output"))
    return {
        "REAL_REGISTRATION_SUPPORTED": "PROVEN" if status == "SUCCESS" and reopened else "DATA_REQUIRED",
        "INDEPENDENT_ACCURACY_SUPPORTED": "PROVEN" if independent_ready else "DATA_REQUIRED",
        "SUBPIXEL_ACCURACY_SUPPORTED": "PROVEN" if independent_ready and result.get("subpixel_count", 0) > 0 else "DATA_REQUIRED",
        "UNIFORM_DISTRIBUTION_SUPPORTED": "PROVEN" if spatial is not None and spatial >= 0.5 else "DATA_REQUIRED",
        "SCALE_ROBUSTNESS_SUPPORTED": "PROVEN" if result.get("scale_ratio") is not None and status == "SUCCESS" else "DATA_REQUIRED",
        "ILLUMINATION_ROBUSTNESS_SUPPORTED": "PROVEN" if result.get("illumination_metadata", {}).get("source") and result.get("illumination_metadata", {}).get("reference") and status == "SUCCESS" else "DATA_REQUIRED",
        "VALIDATION_SCOPE": validation or "DATA_REQUIRED",
        "INLIER_RATIO_AVAILABLE": "AVAILABLE" if inlier_ratio is not None else "NOT_AVAILABLE",
    }
