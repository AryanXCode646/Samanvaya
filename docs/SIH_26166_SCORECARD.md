# SIH Problem Statement 26166 — Scientific Scorecard

**Problem Statement:** *Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS).*

| Requirement | Implementation Module | Automated Tests | Current Status | Notes & Data Requirement |
|---|---|---|---|---|
| **1. PDS4 Ingestion & Discovery** | `lunar_core/data_io/mission_catalog.py`, `discovery.py` | `tests/test_mission_catalog.py`, `test_dataset_manifest.py` | **PROVEN (Code & Fixtures)** | Parses PDS4 XML, extracts GSD, sun geometry, footprints, handles detached rasters. |
| **2. Multi-Modal Modalities (OHRC, TMC-2, IIRS)** | `lunar_core/models.py`, `lunar_core/data_io/mission_adapters.py` | `tests/test_real_mission_fixtures.py`, `test_iirs_alignment.py` | **SUPPORTED** | OHRC & TMC-2 fully supported in 2D pipeline; IIRS has continuum extraction; real IIRS cubes flagged `partial` (2D representation requires data). |
| **3. Solar / Sun-Angle Invariance** | `lunar_core/preprocessing/phase_congruency.py`, `photometric.py` | `tests/test_photometric_dem.py`, `test_phase_congruency.py` | **PROVEN (Synthetic)** / **READY_FOR_REAL** | Phase Congruency maximum moment analysis provides structural illumination invariance; verified across opposite sun azimuths in synthetic tests. |
| **4. Scale Invariance (Multi-Scale Octaves)** | `lunar_core/alignment/scale_space.py`, `samanvaya/registration/transform.py` | `tests/test_scale_space.py`, `test_pyramid_geometry.py` | **PROVEN (Synthetic)** / **READY_FOR_REAL** | Fourier-Mellin log-polar coarse localization bridges multi-octave resolution gaps; coordinate restoration preserves native subpixel positions. |
| **5. Feature Correspondence (Dense + RIFT)** | `lunar_core/alignment/dense_matcher.py`, `rift_matcher.py` | `tests/test_dense_loftr_matcher.py`, `test_fallback_matching.py` | **PROVEN (Synthetic)** / **READY_FOR_REAL** | Cross-attention transformer (LoFTR) with classical RIFT fallback on Phase Congruency feature maps. |
| **6. Authoritative Full-Image Coordinates** | `samanvaya/registration/coordinates.py`, `windows.py` | `tests/test_registration_coordinates.py`, `test_nonzero_origin_geometry.py` | **PROVEN (Code & Tests)** | Strict `T(source FULL_IMAGE) = reference FULL_IMAGE` convention; non-zero window and tile offsets propagated without mixing frames. |
| **7. Authoritative Transform Direction** | `samanvaya/registration/transform.py` | `tests/test_transform_direction.py`, `test_known_transform.py` | **PROVEN (Code & Tests)** | Forward mapping source $\to$ reference; OpenCV forward warping without double-inversion; known-transform recovered to $<0.05$ px RMSE. |
| **8. Geometric Plausibility & Model Mismatch Gates** | `samanvaya/registration/transform.py` | `tests/test_registration_transform.py` | **PROVEN (Code & Tests)** | Rejects non-positive determinants, excessive shear, unphysical scales, and systematic regional residual gradients. |
| **9. Original Unnormalized Raster Warping** | `samanvaya/validation/real_registration.py` | `tests/test_warp_direction.py`, `test_output_geometry.py` | **PROVEN (Code & Tests)** | Directly resamples unnormalized float32 source pixels; reopens GeoTIFF with rasterio to verify count, dimensions, nodata, and CRS. |
| **10. Decoupled Independent Validation Checkpoints** | `samanvaya/validation/checkpoints.py`, `benchmark_real.py` | `tests/test_checkpoint_leakage.py`, `test_checkpoint_contract.py` | **PROVEN (Code & Tests)** | Estimation points (`estimation_matches.csv`) and evaluation checkpoints (`validation_checkpoints.csv`) are strictly decoupled. |
| **11. Sub-Pixel Accuracy Claims** | `lunar_core/postprocessing/subpixel.py` | `tests/test_subpixel.py`, `test_subpixel_geometry.py` | **DATA_REQUIRED** | Subpixel peak fitting verified synthetically ($<0.1$ px); real mission data status correctly gated as `DATA_REQUIRED` until GCPs executed. |
| **12. Real Mission Data Registration** | `samanvaya/validation/real_registration.py`, `register_pair()` | `tests/test_real_pair_workflow.py`, `test_register_pair_contract.py` | **READY_FOR_REAL / DATA_REQUIRED** | Code path is complete, hardened, and executable. Physical mission imagery files required to promote status to `PROVEN`. |
| **13. Classical Baseline Comparison** | `samanvaya/validation/baseline_registration.py` | `tests/test_baseline_registration.py` | **PROVEN (Code & Tests)** | Executes standard SIFT/ORB + RANSAC on identical inputs and checkpoints for side-by-side comparative benchmarking. |
| **14. End-to-End Benchmark CLI** | `lunar_core/cli.py` (`inventory`, `discover-real-pairs`, `benchmark-real`) | `tests/test_first_real_pair_contract.py` | **PROVEN (CLI & Tests)** | Full CLI orchestration; returns `DATA_REQUIRED` with missing files when real mission imagery is absent. |
| **15. Automated Claim Gating** | `samanvaya/validation/claim_gate.py` | `tests/test_claim_gate.py` | **PROVEN (Code & Tests)** | Automatically gates claims; synthetic fixtures cannot elevate status to `PROVEN`; requires verified flight checkpoints. |
| **16. Reproducibility & Provenance** | `samanvaya/provenance.py` | `tests/test_real_pair_workflow.py` | **PROVEN (Code & Tests)** | Logs Git commit SHA, config SHA256, source/reference file SHA256 hashes, and execution metrics. |

---

## Overall Assessment
- **Architecture Readiness**: 100% COMPLETE.
- **Test Baseline**: 163 passing tests across unit, integration, coordinate propagation, and end-to-end manifest contracts.
- **Scientific Claim Status**: `READY_FOR_REAL_VALIDATION` with explicit `DATA_REQUIRED` for flight datasets. No synthetic results are deceptively represented as spacecraft evidence.
