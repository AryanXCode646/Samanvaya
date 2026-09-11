# Full-Image Geometry & Coordinate System Audit

## 1. Executive Summary & Authoritative Convention

Samanvaya establishes an unambiguous, mathematically rigorous coordinate system and transform direction convention across all image registration, tile processing, pyramid resampling, and raster warping pipelines.

### Authoritative Transform Definition
$$\mathbf{T}(\mathbf{p}_{\text{source}}^{\text{FULL\_IMAGE}}) = \mathbf{p}_{\text{reference}}^{\text{FULL\_IMAGE}}$$

Every geometric transformation estimated by Samanvaya maps points from the **source image** (target/unregistered input) into the **reference image** coordinate frame.

- **Source Frame**: `FULL_SOURCE_IMAGE`
- **Target/Reference Frame**: `FULL_REFERENCE_IMAGE`
- **Direction**: `source->reference`
- **Inverse Operation**: $\mathbf{T}^{-1}(\mathbf{p}_{\text{reference}}) = \mathbf{p}_{\text{source}}$ (`reference->source`)

Under no circumstances may bare variable names such as `H`, `M`, `transform`, or `matrix` be used without documenting their directional semantics.

---

## 2. Coordinate Conventions

### Pixel Coordinate Convention
- Coordinate pair: $(x, y) = (\text{column}, \text{row})$
- Origin: $(0.0, 0.0)$ is at the top-left outer corner of pixel $(0, 0)$.
- Pixel center: The center of pixel at row $r$, col $c$ is located at $(c + 0.5, r + 0.5)$ in continuous spatial coordinates. When operating in integer discrete index space, pixel $(c, r)$ corresponds to integer column $c$ and row $r$.
- Axes:
  - $x$ increases horizontally to the right across columns ($0 \le x < \text{width}$).
  - $y$ increases vertically downward across rows ($0 \le y < \text{height}$).
  - Image array shape in NumPy is $(H, W) = (\text{rows}, \text{cols})$.

### Homogeneous Coordinates
Planetary coordinates are represented in projective 2D space:
$$\mathbf{p} = \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

For a projective homography $\mathbf{H} \in \mathbb{R}^{3 \times 3}$:
$$\begin{bmatrix} u \\ v \\ w \end{bmatrix} = \mathbf{H} \begin{bmatrix} x_{\text{source}} \\ y_{\text{source}} \\ 1 \end{bmatrix}$$
$$x_{\text{reference}} = \frac{u}{w}, \quad y_{\text{reference}} = \frac{v}{w} \quad (\text{provided } |w| > 10^{-12})$$

For an affine transformation $\mathbf{A} \in \mathbb{R}^{2 \times 3}$:
$$\begin{bmatrix} x_{\text{reference}} \\ y_{\text{reference}} \end{bmatrix} = \mathbf{A} \begin{bmatrix} x_{\text{source}} \\ y_{\text{source}} \\ 1 \end{bmatrix}$$

---

## 3. Estimation Pipeline & Frame Enforcement

Before estimating global parameters via USAC-MAGSAC++, RANSAC, or Least-Squares:
1. Every source point must belong to `FULL_SOURCE_IMAGE`.
2. Every reference point must belong to `FULL_REFERENCE_IMAGE`.
3. If points from disparate local frames (e.g. `ROI`, `TILE`, `PYRAMID_LEVEL_1`) are passed without explicit restoration to full image, estimation immediately aborts with:
   $$\text{MIXED\_COORDINATE\_FRAMES}$$

### Coordinate Propagation Chain
When dense matching occurs in a downsampled or cropped Region of Interest (ROI):
1. **Matcher Space**: Matcher predicts $(\Delta x, \Delta y)$ within normalized ROI patches.
2. **Common Space Translation**: Add ROI window offsets $(x_{\text{min}}, y_{\text{min}})$.
3. **Coarse Inversion**: Invert coarse alignment affine transformation:
   $$\mathbf{p}_{\text{target\_comm}} = \mathbf{A}_{\text{target\_to\_common}}^{-1} \mathbf{p}_{\text{common}}$$
4. **Resolution Restoration**: Multiply by scale ratios:
   $$x_{\text{full}} = x_{\text{comm}} \times s_{\text{comm\_to\_full}}$$
5. **Full Image Origin**: Add native raster window or tile offset:
   $$x_{\text{source}}^{\text{FULL}} = x_{\text{source\_win}} + \text{col\_off}_{\text{source}}$$
   $$y_{\text{source}}^{\text{FULL}} = y_{\text{source\_win}} + \text{row\_off}_{\text{source}}$$
6. Emit `KeypointMatch` strictly tagged with `source_frame="FULL_SOURCE_IMAGE"` and `reference_frame="FULL_REFERENCE_IMAGE"`.

---

## 4. Raster Warping Convention

### OpenCV Forward vs Inverse Mapping
A frequent source of bugs in remote sensing pipelines is the inverse mapping convention of raster warpers:
- In `cv2.warpAffine` and `cv2.warpPerspective`, when `WARP_INVERSE_MAP` is **NOT** set, OpenCV defines the input matrix $M$ as the **forward mapping from input image to output canvas**:
  $$\text{output}(x, y) = \text{input}\left( M^{-1} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} \right)$$
  OpenCV performs the inversion internally to perform backward pixel lookup with interpolation.
- Therefore, because our transform is defined as $\mathbf{T}(\text{source}) = \text{reference}$, passing $\mathbf{T}$ directly to:
  ```python
  cv2.warpPerspective(source_raster, T, (reference_width, reference_height))
  ```
  correctly projects source pixels into the reference grid.
- **Never invert $\mathbf{T}$ prior to passing to `cv2.warpPerspective` unless `WARP_INVERSE_MAP` is explicitly enabled.** Doing so causes double-inversion and corrupts alignment.

### Original Pixel Integrity
`registered_source.tif` is generated using **original unnormalized raster pixels** read directly from the source mission product, mapped into the reference geometry, and verified upon write with `rasterio`. Preview PNGs or normalized visualization buffers are strictly prohibited in the scientific output path.

---

## 5. Output Georeferencing & Validation

After writing `registered_source.tif`:
- Reference coordinate reference system (CRS, e.g. `IAU2000:30100`) is assigned.
- Reference affine geotransform is preserved:
  $$\text{transform}_{\text{out}} = \text{transform}_{\text{ref}}$$
- Dimensions $(W, H)$ match reference image dimensions.
- `output_validation.json` records:
  - `input_source_transform`: Source raster geotransform
  - `input_reference_transform`: Reference raster geotransform
  - `registration_matrix`: The $3 \times 3$ or $2 \times 3$ pixel-space transformation matrix
  - `crs`: CRS string
  - `bounds`: Bounding box coordinates

---

## 6. Scientific Checkpoint & Leakage Auditing

1. **Independent Evaluation Target**:
   Validation checkpoints are independently surveyed GCPs or ground-truth points. The registered image is an *output artifact*, not the ground-truth target. Reprojection error is calculated directly:
   $$\mathbf{e}_i = \mathbf{p}_i^{\text{ref, true}} - \mathbf{T}(\mathbf{p}_i^{\text{src}})$$
2. **Held-Out Separation**:
   - `estimation_matches.csv`: Tie points used for parameter estimation.
   - `validation_checkpoints.csv`: Strictly held-out checkpoints never exposed to estimator.
3. **Leakage Detection**:
   The reporting system computes both:
   - `fit_residuals` (in-sample error)
   - `held_out_residuals` (out-of-sample error)
   Any discrepancy where fit RMSE is low ($<0.2$ px) but held-out RMSE is large ($>2.0$ px) is flagged to prevent misleading claims.
4. **Plausibility Gates**:
   Transformations with non-positive determinants, scaling factors outside $[0.1, 10.0]$, or extreme shears are flagged as `IMPLAUSIBLE_TRANSFORM`.
5. **Model Mismatch Detection**:
   If residual vectors exhibit coherent spatial tilt or systematic regional bias, registration is flagged as `MODEL_MISMATCH`.
