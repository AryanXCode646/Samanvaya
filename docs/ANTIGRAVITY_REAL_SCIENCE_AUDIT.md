# Antigravity Real Science Audit

## 1. System Architecture & Component Traceability

This document audits the complete scientific path of Samanvaya for Smart India Hackathon Problem Statement 26166:
*"Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS)."*

### End-to-End Geometric & Validation Data Flow
```text
MissionProduct (PDS4 XML/Label or GeoTIFF)
  │
  ▼
Product Discovery & Inventory (scan, discover_benchmark_pairs, inspect_product)
  │
  ▼
Window Extraction (RegistrationWindows with CoordinateAudit col_off, row_off)
  │
  ▼
Photometric & Contrast Normalization (Lommel-Seeliger shading, Phase Congruency max_moment)
  │
  ▼
Multi-Scale ROI Extraction (ScaleSpaceLocalizer with coarse Fourier-Mellin similarity)
  │
  ▼
Feature Correspondence (Dense LoFTR Transformer / Classical RIFT fallback)
  │
  ▼
Coordinate Restoration (Invert coarse target-to-common affine, restore native GSD scale, add window offsets)
  │
  ▼
Frame Integrity Assertion (source_frame == FULL_SOURCE_IMAGE, reference_frame == FULL_REFERENCE_IMAGE)
  │
  ▼
Robust Estimation & Plausibility (USAC-MAGSAC++, LMEDS affine refinement, determinant & shear gates)
  │
  ▼
Structured Transform (RegistrationTransform object: T(source FULL_IMAGE) = reference FULL_IMAGE)
  │
  ▼
Original Raster Warping (cv2.warpPerspective on unnormalized source pixels, write registered_source.tif)
  │
  ▼
Output Georeferencing Verification (Reopen with rasterio, verify width, height, dtype, nodata, CRS, bounds)
  │
  ▼
Held-Out Checkpoint Validation (Strictly decoupled validation_checkpoints.csv vs estimation_matches.csv)
  │
  ▼
Scientific Claim Gate (Automatic evidence evaluation, no synthetic promotions, DATA_REQUIRED preserved)
```

---

## 2. Component Audits

### 2.1 MissionProduct & PDS4 Ingestion
- **Location**: `lunar_core/data_io/mission_product.py`, `lunar_core/data_io/mission_catalog.py`
- **Capabilities**:
  - Ingests PDS4 XML labels (`.xml`, `.lbl`) and raster formats (`.tif`, `.tiff`, `.img`, `.qub`).
  - Extracts spacecraft name, instrument (`CH2_OHRC`, `CH2_TMC2`, `CH2_IIRS`, `LRO_NAC`, `SELENE_TC`), GSD, solar incidence/azimuth angles, CRS, and footprint boundaries.
  - Lifecycle states: `discovered`, `parsed`, `validated`, `partial`, `invalid`, `unsupported`.

### 2.2 Registration & Coordinate Propagation
- **Location**: `samanvaya/registration/coordinates.py`, `samanvaya/registration/transform.py`, `lunar_core/pipeline.py`
- **Conventions**:
  - Authoritative Transform: $\mathbf{T}(\mathbf{p}_{\text{source}}^{\text{FULL\_IMAGE}}) = \mathbf{p}_{\text{reference}}^{\text{FULL\_IMAGE}}$.
  - Continuous pixel coordinates: $(x, y) = (\text{column}, \text{row})$ with top-left origin $(0.0, 0.0)$.
  - Multi-scale pyramid levels ($1.0, 0.5, 0.25, 0.125$) and 2D tile offsets are tracked in `CoordinateAudit`.
  - Mixed coordinate frames are strictly forbidden (`MIXED_COORDINATE_FRAMES`).

### 2.3 Feature Matchers
- **Location**: `lunar_core/alignment/dense_matcher.py`, `lunar_core/alignment/rift_matcher.py`
- **Dense Matcher**: `kornia.feature.LoFTR` cross-attention matcher on phase congruency invariant moments.
- **Classical Fallback**: Classical RIFT (Radiation-Invariant Feature Transform) matching on maximum moment and orientation index maps.

### 2.4 Original Raster Warping & Georeferencing
- **Location**: `samanvaya/validation/real_registration.py`
- **Source Pixels**: `cv2.warpPerspective(source, transform.as_homography(), (ref_w, ref_h))` directly transforms unnormalized native source float32 pixels.
- **Verification**: Reopens output GeoTIFF, checks count, dimensions, nodata, finite pixel presence, and writes `output_validation.json` and `geometry.json`.

### 2.5 Checkpoints & Leakage Protection
- **Location**: `samanvaya/validation/checkpoints.py`, `tests/test_checkpoint_leakage.py`
- **Separation**: `estimation_matches.csv` contains points used for transform fitting; `validation_checkpoints.csv` contains independently surveyed ground truth GCPs.
- **Reporting**: Reports both integer and subpixel RMSE, median, and 95th percentile errors.

### 2.6 Automatic Claim Gating
- **Location**: `samanvaya/validation/claim_gate.py`
- **Rule**: Claims are never promoted to `PROVEN` based on synthetic fixtures or reprojection residuals alone.
- If physical mission data or independent GCPs are missing, the state remains strictly `DATA_REQUIRED`.

---

## 3. Current Test Status

- **Automated Tests**: 159 passed, 1 skipped.
- **Coverage**:
  - Full-image coordinate propagation across windows, pyramids, and tiles.
  - Bidirectional transform application and inversion detection.
  - Synthetic known-transform recovery ($<0.05$ px RMSE).
  - Output GeoTIFF reopening and CRS verification.
  - Checkpoint leakage and integer vs subpixel evaluation.
