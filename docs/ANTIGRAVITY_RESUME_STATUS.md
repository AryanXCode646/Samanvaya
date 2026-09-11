# Antigravity Resume Status & System Audit

**Date**: 2026-09-11  
**Branch**: `hardening-pass`  
**Base Commit**: `eb99db9` (`feat: prove full-image geometry, authoritative transform direction, and coordinate propagation`)  
**Suite Status**: 163 passed, 1 skipped (0 failures, 0 regressions)

---

## 1. Resume Component Status Matrix

| Component | State | Evidence | Next action |
|---|---|---|---|
| **MissionProduct & Ingestion** | **DONE** | `lunar_core/data_io/mission_catalog.py`, `tests/test_mission_catalog.py` | Operational; parses PDS4 XML, extracts GSD and solar angles. |
| **Coordinate Propagation** | **DONE** | `samanvaya/registration/coordinates.py`, `tests/test_registration_coordinates.py`, `test_nonzero_origin_geometry.py` | Preserved; `T(source FULL_IMAGE) = reference FULL_IMAGE`; rejects `MIXED_COORDINATE_FRAMES`. |
| **Transform Direction & Plausibility** | **DONE** | `samanvaya/registration/transform.py`, `tests/test_transform_direction.py`, `test_registration_transform.py` | Enforces forward transform, determinant $> 0$, condition number, and spatial model mismatch detection. |
| **Original Raster Warping** | **DONE** | `samanvaya/validation/real_registration.py`, `tests/test_warp_direction.py`, `test_output_geometry.py` | Directly warps unnormalized native source float32 pixels; reopens and validates GeoTIFF with rasterio. |
| **Checkpoint Leakage Protection** | **DONE** | `samanvaya/validation/checkpoints.py`, `tests/test_checkpoint_leakage.py` | Strictly decouples `estimation_matches.csv` from held-out `validation_checkpoints.csv`. |
| **Authoritative `register_pair()` API** | **DONE** | `samanvaya/validation/real_registration.py`, `tests/test_register_pair_contract.py` | Standardized API and complete properties on `RealRegistrationResult`. |
| **Classical Baseline Comparison** | **DONE** | `samanvaya/validation/baseline_registration.py`, `tests/test_baseline_registration.py` | Implemented SIFT/ORB + RANSAC baseline for direct comparison on identical inputs and checkpoints. |
| **Visual Evidence & Output Package** | **DONE** | `samanvaya/validation/real_registration.py`, `visual/` directory generation | Generates complete package (`source_metadata.json`, `reference_metadata.json`, `overlap.json`, `provenance.json`, `coordinate_audit.json`, `spatial_distribution.json`, visual previews, and `failure.json`). |
| **CLI Tools (`inventory`, `discover-real-pairs`)** | **DONE** | `lunar_core/cli.py` (`--output` made optional) | Verified: `./.venv/bin/python -m lunar_core.cli inventory --root data` runs cleanly. |
| **Real Benchmark Execution (`benchmark-real`)** | **DONE** | `samanvaya/validation/benchmark_real.py`, `evidence/real_data_manifest.json` | Verified: returns structured `DATA_REQUIRED` with explicit missing resources when rasters are absent. |
| **Claim Gating** | **DONE** | `samanvaya/validation/claim_gate.py`, `tests/test_claim_gate.py` | Evaluates evidence without synthetic promotion; physical data required for `PROVEN`. |
| **IIRS Support** | **PARTIAL** | `lunar_core/preprocessing/spectral.py`, `scripts/register_real_pair.py` | Continuum extraction code exists and tested synthetically; real multi-band cubes remain `partial`/`DATA_REQUIRED`. |
| **Real Spacecraft Flight Data** | **DATA_REQUIRED** | `evidence/real_data_manifest.json` | Physical flight products (CH2 OHRC/TMC-2/IIRS and LRO NAC) not present in local checkout. |

---

## 2. Recovery & Interruption Analysis

### What Was Already Completed Before Interruption
- Full-image coordinate restoration across pyramid downsampling, 2-D tiles, and non-zero window origins.
- Authoritative transform mapping definition: $\mathbf{T}(\mathbf{p}_{\text{source}}^{\text{FULL\_IMAGE}}) = \mathbf{p}_{\text{reference}}^{\text{FULL\_IMAGE}}$.
- Rejection of mixed coordinate frames prior to geometric estimation (`MIXED_COORDINATE_FRAMES`).
- `RegistrationTransform` dataclass with geometric plausibility gates (determinant sign, scale bounds, shear).
- Decoupled checkpoint validation protecting against evaluation leakage.
- 10 dedicated geometric test suites passing 159 tests.

### What Was Incomplete / Broken
1. **CLI Argument Inflexibility**: `lunar_core.cli inventory` and `discover-real-pairs` required `--output` as a mandatory flag, causing execution failure when invoked without `--output`.
2. **Benchmark Manifest Real-Data Pairing**: `evidence/real_data_manifest.json` lacked explicit benchmark pairs matching `schemas/real_benchmark_manifest.schema.json`.
3. **Benchmark Overall Status**: `samanvaya.validation.benchmark_real.run_real_benchmark` returned `"status": "COMPLETE"` even when all rows were `"DATA_REQUIRED"` due to missing flight rasters.
4. **Authoritative `register_pair()` Alias & Properties**: Missing contract properties (`source_product_id`, `reference_product_id`, `selected_match_count`, `transform`, `subpixel_statistics`, `match_output`) on `RealRegistrationResult`.
5. **Classical Baseline Comparative Path**: Missing baseline registration pipeline (SIFT/ORB + RANSAC) to compare directly against Samanvaya on identical inputs.
6. **Complete Real Output Package**: Missing automatic export of `source_metadata.json`, `reference_metadata.json`, `overlap.json`, `provenance.json`, `coordinate_audit.json`, `spatial_distribution.json`, and visual thumbnails.
7. **Documentation Protocol & Scorecard**: Missing `docs/REAL_EXPERIMENT_PROTOCOL.md` and `docs/SIH_26166_SCORECARD.md`.

### What Was Completed & Fixed
1. **CLI Hardened**: Made `--output` optional for `inventory` and `discover-real-pairs` so both work seamlessly with or without output file redirection.
2. **Candidate Benchmark Pairs Defined**: Populated `evidence/real_data_manifest.json` with the canonical candidate pairs (`CH2_OHRC__LRO_NAC__SITE_01`, `CH2_TMC2__LRO_NAC__SITE_01`, `CH2_IIRS__LRO_NAC__SITE_01`), validating both schema compliance and backward-compatibility with `tests/test_validation_framework.py`.
3. **Explicit `DATA_REQUIRED` Semantics**: Updated `benchmark_real.py` so that when rasters are absent, the overall benchmark result correctly reports `status: DATA_REQUIRED` with explicit paths to missing resources.
4. **Contract Properties & `register_pair`**: Added all Section 10 properties and aliases to `RealRegistrationResult` and exported `register_pair()`.
5. **Classical Baseline Implemented**: Created `samanvaya/validation/baseline_registration.py` executing standard SIFT/ORB + RANSAC and evaluating against independent checkpoints.
6. **Full Output Package & Visuals**: Extended `register_products()` to emit all 12 metadata, audit, and diagnostic files plus visual thumbnails (`source.png`, `reference.png`, `raw_matches.png`, `verified_matches.png`, `uniform_matches.png`, `registered_overlay.png`, `residual_vectors.png`).
7. **Protocol & Scorecards Created**: Generated `docs/REAL_EXPERIMENT_PROTOCOL.md`, `docs/SIH_26166_SCORECARD.md`, and this resume status audit.
8. **Automated Tests Added**: Added `tests/test_baseline_registration.py` and `tests/test_register_pair_contract.py`, bringing the passing test count to 163.

---

## 3. What the Tests Prove vs What Remains DATA_REQUIRED

### What the Tests Prove
- Mathematical correctness of full-image coordinate propagation across multi-scale pyramids, non-zero image windows, and tile grids.
- Inverse-free OpenCV warping of original unnormalized source rasters into the reference frame.
- Known-transform recovery to $<0.05$ px RMSE on synthetic fixtures.
- Checkpoint leakage isolation and distinction between integer and sub-pixel residuals.
- Ingestion and metadata validation of PDS4 XML labels and detached rasters.
- Fail-safe CLI and manifest orchestration returning `DATA_REQUIRED` without fabricating fake data.

### What Remains DATA_REQUIRED
- Actual physical flight rasters for Chandrayaan-2 OHRC, TMC-2, and IIRS paired with LRO NAC reference images.
- Authorized, independently surveyed ground control point networks (GCPs) on actual lunar terrain.
- Promotion of `SUBPIXEL_ACCURACY`, `REAL_REGISTRATION`, and `CROSS_MODAL_REGISTRATION` to `PROVEN` on spacecraft data.
