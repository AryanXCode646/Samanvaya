# Planetary and Image Coordinate Systems Specification
## Samanvaya Framework (SIH PS 26166)

This document authoritatively specifies all coordinate conventions, origin conventions, pixel center definitions, and planetary reference frames used in Samanvaya.

---

### 1. Pixel Coordinate Convention
* **Coordinate Type:** Continuous floating-point representation `(x, y)` in `float64` / `float32`.
* **Axis Ordering:** 
  * `x`: Horizontal axis, column index, sample direction ($0 \le x < W$).
  * `y`: Vertical axis, row index, line direction ($0 \le y < H$).
* **Array Indexing:**
  * NumPy arrays are indexed as `image[row, col]` = `image[int(round(y)), int(round(x))]`.
  * Spatial height $H = \text{image.shape}[0]$, spatial width $W = \text{image.shape}[1]$.
* **Pixel Center Convention:**
  * Coordinates are 0-indexed.
  * The center of pixel `(col, row)` is at `(col + 0.5, row + 0.5)` for continuous integration, or integer coordinates `(col, row)` represent sample indices. Sub-pixel refinements $dx \in [-1.0, 1.0]$ and $dy \in [-1.0, 1.0]$ modify continuous continuous locations around the integer sample grid.

---

### 2. Image Origin & Orientation
* **Origin `(0.0, 0.0)`:** Upper-left corner of the raster array.
* **X Direction:** Increases to the right (Eastward in standard cartographic map projections).
* **Y Direction:** Increases downward (Southward in standard cartographic map projections).

---

### 3. Coordinate Frames
* **`FULL_SOURCE_IMAGE`:** Native full-resolution image coordinate frame of the source / moving product (e.g. Chandrayaan-2 OHRC at 0.25 m/px).
* **`FULL_REFERENCE_IMAGE`:** Native full-resolution image coordinate frame of the reference / fixed product (e.g. LRO NAC at 0.50 m/px or TMC-2 at 5.0 m/px).
* **`COMMON_ROI_FRAME`:** Intermediate resampled coordinate frame used during coarse scale-space alignment. All matcher coordinates extracted on ROI tiles are strictly mapped back to `FULL_SOURCE_IMAGE` and `FULL_REFERENCE_IMAGE` before ANMS capping and transformation estimation.
* **Transformation Direction:**
  $$\mathbf{x}_{\text{ref}} \sim \mathbf{H} \cdot \mathbf{x}_{\text{src}}$$
  Where $\mathbf{x}_{\text{src}} = [x_{\text{src}}, y_{\text{src}}, 1]^T$ in `FULL_SOURCE_IMAGE` frame, and $\mathbf{x}_{\text{ref}} = [x_{\text{ref}}, y_{\text{ref}}, 1]^T$ in `FULL_REFERENCE_IMAGE` frame.

---

### 4. Planetary Coordinate Reference System (CRS)
* **Target Body:** Moon (Earth's Moon).
* **Datum / Ellipsoid:** IAU/IAG 2015 Lunar Reference System (`IAU_2015_MOON` / `IAU2000:30100`).
* **Mean Lunar Radius:** $R_{\text{Moon}} = 1737.4\text{ km}$.
* **Latitude Convention:** Planetocentric latitude $\phi \in [-90.0^\circ, +90.0^\circ]$, with $0^\circ$ at lunar equator and $+90^\circ$ at North Pole.
* **Longitude Convention:** Positive East $\lambda \in [-180.0^\circ, +180.0^\circ]$ (or $[0^\circ, 360.0^\circ]$). Antimeridian crossing ($\pm 180^\circ$) is explicitly unwrapped before spherical distance and polygon intersection operations.

---

### 5. Provenance Requirements
Every quantitative result produced by Samanvaya must record:
1. `metric_basis`: Whether metrics are derived from independent ground-truth (`ground_truth_control_points`) or purely from `reprojection_consensus`.
2. `source_frame` and `reference_frame`: Explicit coordinate system strings.
3. `software_version` and `git_commit_sha`: Exact commit and execution environment.
4. `scientific_status`: Must be one of:
   * `GROUND_TRUTH_VALIDATED`
   * `REPROJECTION_ONLY`
   * `NO_GROUND_TRUTH`
   * `FAILED_VALIDATION`
   * `CANDIDATE_UNVERIFIED`
   * `PROXIMITY_CANDIDATE`
