# Real Mission Data Experiment Protocol

## 1. Scope & Objective
This protocol establishes the rigorous, scientific, and reproducible testing methodology for multi-modal optical lunar image registration under Smart India Hackathon Problem Statement 26166:
*"Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS)."*

The protocol enforces strict separation between:
1. **Model Parameter Estimation Tie Points** (`estimation_matches.csv`)
2. **Independently Surveyed Evaluation Checkpoints** (`validation_checkpoints.csv`)

Synthetic fixtures, demo datasets, and uncalibrated visual checks are explicitly disqualified from supporting scientific claims.

---

## 2. Dataset Definitions & Hierarchy

### 2.1 Primary Lunar Datasets
| Mission | Instrument | Native GSD | Spectral Bandpass / Description | Radiometric Format |
|---|---|---|---|---|
| **Chandrayaan-2** | OHRC (Optical High Resolution Camera) | $0.25 - 0.32\text{ m/px}$ | $450 - 900\text{ nm}$ (Panchromatic) | PDS4 8-bit/16-bit raster (`.img`, `.tif`) + XML label |
| **Chandrayaan-2** | TMC-2 (Terrain Mapping Camera 2) | $5.0\text{ m/px}$ | $500 - 850\text{ nm}$ (Stereo: Fore, Nadir, Aft) | PDS4 raster (`.img`, `.tif`) + XML label |
| **Chandrayaan-2** | IIRS (Imaging Infrared Spectrometer) | $80.0\text{ m/px}$ | $0.8 - 5.0\ \mu\text{m}$ (256 spectral bands) | PDS4 spectral cube (`.qub`, `.img`) + XML label |
| **LRO** (Reference) | LROC NAC (Narrow Angle Camera) | $0.50\text{ m/px}$ | $400 - 750\text{ nm}$ (Panchromatic) | PDS3/PDS4 GeoTIFF (`.tif`) or calibrated cube |
| **SELENE** (Reference) | TC (Terrain Camera) | $10.0\text{ m/px}$ | $430 - 850\text{ nm}$ (Stereo Nadir/Aft) | PDS3 / GeoTIFF |

### 2.2 Storage & File Paths
All physical flight rasters must reside in structured paths:
```text
data/
  missions/
    chandrayaan2/
      ohrc/
        <product_id>.img
        <product_id>.xml
      tmc/
        <product_id>.tif
        <product_id>.xml
      iirs/
        <product_id>.qub
        <product_id>.xml
  reference/
    lro_nac/
      <product_id>.tif
      <product_id>.xml
  checkpoints/
    <pair_id>.csv
```

---

## 3. Candidate Pair Selection & Screening

A candidate pair $(\mathcal{P}_{\text{source}}, \mathcal{P}_{\text{reference}})$ is eligible for registration only if it passes all pre-screening gates:

1. **Metadata Verification**:
   - Both products must successfully parse via `inspect_product()`.
   - Ingestion status must be `validated` (or `partial` for spectral cubes undergoing continuum extraction).
   - Valid ground sample distance ($GSD > 0$) must be present in metadata or derived from affine raster transform.

2. **Spatial Footprint Overlap**:
   - Geographic overlap between polygon/bounding box footprints must be verified or marked `APPROXIMATE` / `EXACT`.
   - If footprints indicate disjoint regions, execution is rejected with `NO_OVERLAP`.
   - At least $256 \times 256$ pixels of spatial overlap must exist within the intersecting window (`NO_PIXEL_OVERLAP`).

3. **Illumination Angle Recording**:
   - Solar azimuth and elevation/incidence angles must be logged from PDS4 observational geometry.
   - If solar geometry is absent, default synthetic angles are **strictly prohibited**; missing angles are recorded as `unavailable`.

---

## 4. Authoritative Registration Methodology

1. **Window Extraction**:
   - Native raster boundaries are read without downsampling or destructive re-encoding using `rasterio.windows`.
   - Integer window offsets $(\text{col\_off}, \text{row\_off})$ are preserved in `CoordinateAudit`.

2. **Structural Illumination Invariance**:
   - Apply Phase Congruency maximum moment analysis ($M_{\text{max}}$) across 4 scales and 6 orientations:
     $$PC(x, y) = \frac{\sum_o \max(0, E_o(x, y) - T_o)}{\sum_o \sum_n A_{n, o}(x, y) + \epsilon}$$
   - Energy values are invariant to monotonic illumination gradients, shadow reversals, and sensor gain disparities.

3. **Scale Space & Coarse Alignment**:
   - Scale ratio $S = \max(GSD_{\text{src}}, GSD_{\text{ref}}) / \min(GSD_{\text{src}}, GSD_{\text{ref}})$.
   - Resample coarse image to equal ground resolution; determine coarse similarity transform $\mathbf{A}_{\text{coarse}}$ via Fourier-Mellin correlation.

4. **Dense Correspondence & Coordinate Restoration**:
   - High-confidence feature correspondence extraction via cross-attention transformer (`LoFTR`) or classical `RIFT` fallback.
   - Every tie point coordinate must be explicitly restored to full-image coordinate space:
     $$\mathbf{p}_{\text{source}}^{\text{FULL}} = \mathbf{A}_{\text{coarse}}^{-1}(\mathbf{p}_{\text{coarse}}) \times s + \begin{bmatrix} \text{col\_off} \\ \text{row\_off} \end{bmatrix}_{\text{source}}$$
   - Target frame must be `FULL_REFERENCE_IMAGE` and source frame must be `FULL_SOURCE_IMAGE`.

5. **Robust Geometric Estimation**:
   - USAC-MAGSAC++ robust estimation with minimum 4 non-collinear correspondences.
   - Least-squares affine inlier refinement (LMEDS) with residual thresholding.

6. **Transform Convention**:
   $$\mathbf{T}(\mathbf{p}_{\text{source}}^{\text{FULL\_IMAGE}}) = \mathbf{p}_{\text{reference}}^{\text{FULL\_IMAGE}}$$
   OpenCV raster warp maps native unnormalized source pixels into reference grid:
   ```python
   cv2.warpPerspective(source, transform.as_homography(), (ref_w, ref_h))
   ```

---

## 5. Independent Ground-Truth Checkpoint Methodology

To prevent overfitting and circular validation:
1. **Total Decoupling**:
   - Parameter fitting uses `estimation_matches.csv`.
   - Accuracy verification uses `validation_checkpoints.csv`.
   - Checkpoint coordinates are never seen by the matcher, RANSAC, or refinement optimizers.

2. **Checkpoint Format**:
   ```csv
   point_id,source_x,source_y,reference_x,reference_y,provenance,quality
   cp_001,1045.25,2301.80,1052.10,2298.40,USGS_LROC_GCP_NETWORK,A
   ```

3. **Metrics Computed**:
   - Root Mean Square Error (RMSE):
     $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N \|\mathbf{T}(\mathbf{p}_i^{\text{src}}) - \mathbf{p}_i^{\text{ref}}\|^2}$$
   - Median residual error (50th percentile)
   - 95th percentile residual error (P95)
   - Maximum error
   - Mean residual error

---

## 6. Configurable Thresholds & Failure Semantics

Universal, arbitrary thresholds are rejected in favor of mission-instrument tailored gates:

| Parameter | Default Threshold | Rationale | Failure Status |
|---|---|---|---|
| Min Inlier Count | $\ge 15$ | Required for overdetermined projective stability (8 DOF) | `LOW_INLIER_COUNT` |
| Min Inlier Ratio | $\ge 0.20$ | Below 20% inliers indicates high probability of false matches | `LOW_INLIER_RATIO` |
| Spatial Coverage Entropy | $\ge 0.40$ | Matches clustered in one corner produce degenerate transforms elsewhere | `LOW_SPATIAL_COVERAGE` |
| Scale Ratio Bound | $[0.1, 10.0]$ | Physical magnification beyond 10x requires multi-tier scale bridging | `IMPLAUSIBLE_TRANSFORM` |
| Transform Determinant | $> 0$ | Non-positive determinant indicates reflection or folding | `IMPLAUSIBLE_TRANSFORM` |
| Subpixel RMSE Claim Gate | $< 0.50\text{ px}$ | Independent held-out RMSE must be sub-pixel to claim `PROVEN` | `CHECKPOINT_VALIDATION_FAILED` |

### Failure Code Hierarchy
- `DATA_REQUIRED`: Authorized flight products or GCPs missing from disk.
- `INVALID_SOURCE` / `INVALID_REFERENCE`: Corrupt PDS4 labels or unreadable rasters.
- `NO_OVERLAP` / `NO_PIXEL_OVERLAP`: Imagery does not cover the same lunar terrain.
- `NO_CORRESPONDENCE`: Feature matcher found zero matches.
- `LOW_INLIER_RATIO`: Matches failed RANSAC geometric consensus.
- `IMPLAUSIBLE_TRANSFORM`: Transform exhibits negative determinant or unphysical shear.
- `MODEL_MISMATCH`: Homography cannot accommodate severe relief displacement.
- `WARP_FAILURE`: Memory or dimensions error during resampling.
- `OUTPUT_VERIFICATION_FAILED`: Written GeoTIFF cannot be reopened or contains all-nodata pixels.
- `CHECKPOINT_VALIDATION_FAILED`: Held-out checkpoints exceed error threshold.

---

## 7. Provenance & Reproducibility Package

For every execution, the system emits an immutable run package in `output/real/<pair_id>/`:
```text
output/real/<pair_id>/
  ├── source_metadata.json          # Extracted source PDS4 metadata
  ├── reference_metadata.json       # Extracted reference PDS4 metadata
  ├── overlap.json                  # Footprint & pixel overlap audit
  ├── provenance.json               # Git SHA, config hash, source/ref SHA256
  ├── coordinate_audit.json         # Frame conventions, native window offsets
  ├── geometry.json                 # Matrix parameters, condition number, inliers
  ├── matches.csv / matches.json    # All candidate correspondences
  ├── estimation_matches.csv        # Filtered inliers used for estimation
  ├── registered_source.tif         # Unnormalized warped source GeoTIFF
  ├── output_validation.json        # Reopened raster properties & bounds
  ├── held_out_metrics.json         # Independent checkpoint metrics (if GCPs supplied)
  ├── spatial_distribution.json     # Uniformity entropy & coverage fractions
  └── visual/
      ├── source.png                # Source thumbnail
      ├── reference.png             # Reference thumbnail
      ├── raw_matches.png           # Match visualization
      ├── verified_matches.png      # Inlier visualization
      ├── uniform_matches.png       # Reference-frame scatter plot
      ├── registered_overlay.png    # Composite alignment preview
      └── residual_vectors.png      # Quiver plot of residual displacements
```
