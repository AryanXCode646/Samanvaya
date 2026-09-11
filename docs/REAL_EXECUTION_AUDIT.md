# Real Execution Audit

## Current executable path

| Stage | Input | Module | Output / coordinate semantics |
|---|---|---|---|
| Product inspection | PDS4/GeoTIFF path | `mission_catalog.inspect_product` | `MissionProduct`; identity, dimensions, GSD, sun metadata, footprint status |
| Window extraction | `MissionProduct` pair | `samanvaya.registration.windows.extract_registration_windows` | Rasterio windows, masks, transforms, native full-image offsets, `OVERLAP_UNKNOWN` unless authoritative overlap exists |
| Scale derivation | Product GSD or affine transform resolution | `real_registration.register_products` | Source/reference GSD and ratio; invalid scale blocks registration |
| Illumination metadata | MissionProduct sun fields | `real_registration.register_products` | `SunAngles` only when both values are present; otherwise recorded unavailable |
| Representation | Native 2-D arrays | `LunarCorePipeline.register` | percentile-normalized fallback or photometric normalization, followed by phase congruency |
| Coarse localization | normalized arrays, GSD | `ScaleSpaceLocalizer.extract_coarse_roi` | resampled ROI; current pipeline ROI coordinates are local to the ROI and require audit when tiled/windowed |
| Matcher | ROI arrays | `DenseTransformerMatcher` / RIFT fallback | `KeypointMatch` candidates |
| Spatial selection | candidate matches | `SpatialUniformDistributor` | capped grid-selected matches |
| Geometry | selected matches | `RobustEstimator` | affine/projective matrix and inliers |
| Sub-pixel refinement | verified inliers and phase maps | `AnalyticalSubpixelRefiner` | refined inliers; integer coordinates remain available in match objects |
| Warp | original source array and transform | `cv2.warpPerspective` in `real_registration` | original-source registered raster in reference dimensions |
| Output verification | registered GeoTIFF | `_write_registered` / `verify-output` | reopen, dimensions, valid pixels, dtype, CRS, nodata |
| Checkpoint validation | independent checkpoint file + transform | `checkpoints.py`, `benchmark_real.py` | held-out metrics; never fitting-point ground truth |
| Claim gating | structured result | `claim_gate.evaluate_claims` | `PROVEN` only when evidence fields exist, otherwise `DATA_REQUIRED` |

## Known coordinate risks

- The coarse ROI now carries the target-to-common affine and separate common-grid scale factors. Matcher target points are inverse-transformed back to the original target image before geometry; tiled/windowed callers must still supply their non-zero parent origins through the same frame abstraction.
- Scale-space resampling uses `cv2.resize`; the selected scale and resampling factors must remain in provenance.
- The real output is warped from the original source array, while matching uses normalized representations. This is intentional, but the representation and normalization must be recorded.
- Metadata geographic footprint status is not pixel overlap. The window extractor reports `OVERLAP_UNKNOWN` unless authoritative overlap geometry is available.

## Removed/guarded real-path risks

- No hard-coded solar angles are used by `real_registration.register_products`.
- No random/synthetic raster fallback is used.
- No identity transform is emitted when matching fails.
- Missing/invalid GSD blocks registration unless a usable affine raster resolution exists.
- Registered output is reopened before a successful result is returned.
