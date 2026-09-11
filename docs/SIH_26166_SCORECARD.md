# SIH Problem Statement 26166 — Scientific Scorecard

**Problem Statement:** *Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS).*

---

## 1. Core Requirements Traceability Matrix

| Requirement | IMPLEMENTED | SYNTHETICALLY TESTED | REAL-DATA EXECUTED | INDEPENDENTLY VALIDATED | PROVEN Status | Scientific Notes & Data Dependency |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **1. Multi-Modal** (OHRC, TMC-2, IIRS ↔ LRO NAC, SELENE) | **YES** | **YES** | **PARTIAL** | **NO** | `SUPPORTED` / `DATA_REQUIRED` | Ingestion, 2-D panchromatic, and multi-band continuum extraction implemented; IIRS multi-band cubes gated with `IIRS_REPRESENTATION_UNCERTAIN` unless calibrated; flight rasters required for full proof. |
| **2. Sun-Angle Invariance** (Opposite illumination, $180^\circ$ shadow reversal) | **YES** | **YES** | **NO** | **NO** | `SUPPORTED (Synthetic)` / `DATA_REQUIRED` | Phase Congruency maximum moment feature extraction invariant to solar DC offset; synthetically evaluated across solar variation ($12^\circ \leftrightarrow 65^\circ$) and $180^\circ$ shadow inversions; flight validation pending real mission pairs. |
| **3. Scale Invariance** (Multi-octave scale bridging, up to $320\times$) | **YES** | **YES** | **NO** | **NO** | `SUPPORTED (Synthetic)` / `DATA_REQUIRED` | Fourier-Mellin log-polar coarse localization and multi-resolution pyramid bridge octave gaps; capability synthetically evaluated, but real OHRC↔LRO scale robustness remains pending real-data execution. |
| **4. Sub-Pixel Accuracy** ($\text{RMSE} < 0.40\text{ px}$) | **YES** | **YES** | **NO** | **NO** | `SUPPORTED (Synthetic)` / `DATA_REQUIRED` | Continuous parabolic Taylor Hessian refinement ($<0.10\text{ px}$ synthetic); claim gate strictly enforces independent held-out checkpoints before declaring `PROVEN`. |
| **5. Uniform Distribution** (Non-clumping on source image) | **YES** | **YES** | **NO** | **NO** | `SUPPORTED / SYNTHETICALLY PROVEN` / `READY_FOR_REAL` | Algorithmic enforcement of distribution constraint in `FULL_SOURCE_IMAGE` coordinates is synthetically proven and regression-tested; real flight distribution is pending real flight imagery. |
| **6. Registered Product** (Original unnormalized GeoTIFF export) | **YES** | **YES** | **NO** | **NO** | `PROVEN (Code & Synthetic)` / `READY_FOR_REAL` | Directly resamples unnormalized float32 source raster; writes `registered_source.tif` and independently reopens with rasterio to verify count, dimensions, nodata, and finite data. |
| **7. Match Points** (Authoritative coordinates & ISIS3 GCPs) | **YES** | **YES** | **NO** | **NO** | `PROVEN (Code & Synthetic)` / `READY_FOR_REAL` | Exports `matches.csv` and `matches.json` with explicit `source_full_x`, `source_full_y`, `reference_full_x`, `reference_full_y`; native USGS ISIS3 jigsaw format supported. |
| **8. Evaluation Metrics** (Reprojection RMSE, CE90, Spatial Entropy) | **YES** | **YES** | **NO** | **NO** | `PROVEN (Code & Synthetic)` / `READY_FOR_REAL` | Full diagnostic engine computing RMSE, mean/median residual, CE90, normalized Shannon entropy $H$, and integer vs subpixel held-out delta. |
| **9. Real-Data Validation** (Authoritative mission benchmark) | **YES** | **YES** | **NO** | **NO** | `DATA_REQUIRED` | Pipeline, inventory, and benchmark runner fully hardened; transparently reports `DATA_REQUIRED` when physical flight rasters are not present locally. |
| **10. Reproducibility & Provenance** (Bit-exact audit trail) | **YES** | **YES** | **YES** | **YES** | `PROVEN` | Logs streaming SHA-256 file hashes, config hash, Git commit SHA, Python environment, transform parameters, and audit records. |

---

## 2. Claim-Gate Status Summary

* **Real Registration**: `DATA_REQUIRED` (Code complete; awaiting authorized mission raster checkout).
* **Cross-Modal Registration**: `SUPPORTED` (Implemented; spectral and resolution bridging operational).
* **Scale Robustness**: `SUPPORTED` (Implemented and synthetically evaluated; capability bridge).
* **Illumination Robustness**: `SUPPORTED` (Phase Congruency zero-DC moment analysis synthetically evaluated).
* **Uniform Distribution**: `SUPPORTED / SYNTHETICALLY PROVEN / READY_FOR_REAL`.
* **Subpixel Accuracy**: `DATA_REQUIRED` (Gated until independent mission ground control points evaluated).
* **Independent Validation**: `DATA_REQUIRED` (Evaluator requires physical held-out checkpoints).

---

## 3. Overall Verification Assessment

* **Test Suite**: 167 passed, 1 skipped.
* **Code Integrity**: Zero compiler errors, zero diff violations, coordinate and transform direction verified.
* **Scientific Honesty**: No synthetic results are deceptively represented as orbital spacecraft evidence.
