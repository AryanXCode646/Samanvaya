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
    subpixel_rmse = held_out.get("subpixel_rmse_pixels") or held_out.get("rmse_pixels")
    improvement = held_out.get("improvement_percentage", 0.0)
    spatial = result.get("coverage_fraction")
    inlier_ratio = result.get("inlier_ratio")
    reopened = bool(result.get("registered_output"))
    scale_ratio = result.get("scale_ratio")
    illum = result.get("illumination_metadata") or {}
    has_illum = bool(illum.get("source") and illum.get("reference"))
    modality = str(result.get("provenance", {}).get("modality_classification", "CROSS_MODALITY"))

    # Canonical Phase 24 claim evaluations
    if status == "SUCCESS" and reopened:
        real_reg = "PROVEN"
    elif status in {"IMPLAUSIBLE_TRANSFORM", "NO_CORRESPONDENCE", "DEGENERATE_TRANSFORM", "WARP_FAILURE", "OUTPUT_VERIFICATION_FAILED"}:
        real_reg = "FAILED"
    elif status == "PARTIAL":
        real_reg = "PARTIAL"
    else:
        real_reg = "DATA_REQUIRED"

    if independent_ready:
        indep_val = "PROVEN"
    elif held_out.get("status") == "CHECKPOINT_VALIDATION_FAILED":
        indep_val = "FAILED"
    else:
        indep_val = "DATA_REQUIRED"

    if independent_ready and subpixel_rmse is not None and subpixel_rmse < 0.50:
        subpixel = "PROVEN"
    elif independent_ready and subpixel_rmse is not None and subpixel_rmse < 1.0:
        subpixel = "SUPPORTED"
    elif independent_ready:
        subpixel = "FAILED"
    else:
        subpixel = "DATA_REQUIRED"

    if spatial is not None and spatial >= 0.5 and status == "SUCCESS":
        uniform = "PROVEN"
    elif spatial is not None and spatial >= 0.2 and status == "SUCCESS":
        uniform = "PARTIAL"
    elif status == "SUCCESS":
        uniform = "FAILED"
    else:
        uniform = "DATA_REQUIRED"

    if status == "SUCCESS" and scale_ratio is not None and scale_ratio >= 2.0 and independent_ready:
        scale_claim = "PROVEN"
    elif status == "SUCCESS" and scale_ratio is not None:
        scale_claim = "SUPPORTED"
    elif status in {"IMPLAUSIBLE_TRANSFORM", "NO_CORRESPONDENCE"}:
        scale_claim = "FAILED"
    else:
        scale_claim = "DATA_REQUIRED"

    if status == "SUCCESS" and has_illum and independent_ready:
        illum_claim = "PROVEN"
    elif status == "SUCCESS" and has_illum:
        illum_claim = "SUPPORTED"
    elif status in {"IMPLAUSIBLE_TRANSFORM", "NO_CORRESPONDENCE"}:
        illum_claim = "FAILED"
    else:
        illum_claim = "DATA_REQUIRED"

    if status == "SUCCESS" and modality in {"CROSS_MODALITY", "SPECTRAL_TO_2D", "MULTIRESOLUTION"} and independent_ready:
        cross_modal = "PROVEN"
    elif status == "SUCCESS" and modality in {"CROSS_MODALITY", "SPECTRAL_TO_2D", "MULTIRESOLUTION"}:
        cross_modal = "SUPPORTED"
    elif status in {"IMPLAUSIBLE_TRANSFORM", "NO_CORRESPONDENCE"}:
        cross_modal = "FAILED"
    else:
        cross_modal = "DATA_REQUIRED"

    return {
        # Canonical Phase 24 claim keys
        "REAL_REGISTRATION": real_reg,
        "CROSS_MODAL_REGISTRATION": cross_modal,
        "SCALE_ROBUSTNESS": scale_claim,
        "ILLUMINATION_ROBUSTNESS": illum_claim,
        "UNIFORM_DISTRIBUTION": uniform,
        "SUBPIXEL_ACCURACY": subpixel,
        "INDEPENDENT_VALIDATION": indep_val,
        # Backwards-compatible supported keys
        "REAL_REGISTRATION_SUPPORTED": "PROVEN" if status == "SUCCESS" and reopened else "DATA_REQUIRED",
        "INDEPENDENT_ACCURACY_SUPPORTED": "PROVEN" if independent_ready else "DATA_REQUIRED",
        "SUBPIXEL_ACCURACY_SUPPORTED": "PROVEN" if independent_ready and result.get("subpixel_count", 0) > 0 else "DATA_REQUIRED",
        "UNIFORM_DISTRIBUTION_SUPPORTED": "PROVEN" if spatial is not None and spatial >= 0.5 else "DATA_REQUIRED",
        "SCALE_ROBUSTNESS_SUPPORTED": "PROVEN" if scale_ratio is not None and status == "SUCCESS" else "DATA_REQUIRED",
        "ILLUMINATION_ROBUSTNESS_SUPPORTED": "PROVEN" if has_illum and status == "SUCCESS" else "DATA_REQUIRED",
        "VALIDATION_SCOPE": validation or "DATA_REQUIRED",
        "INLIER_RATIO_AVAILABLE": "AVAILABLE" if inlier_ratio is not None else "NOT_AVAILABLE",
    }
