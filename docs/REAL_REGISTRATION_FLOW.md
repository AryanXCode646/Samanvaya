# Real Registration Flow

The real path is intentionally separate from bundled synthetic benchmarks.

1. **Manifest / discovery**
   - `lunar_core.data_io.discovery.discover_products`
   - `lunar_core.data_io.mission_catalog.inspect_product`
   - Output: `MissionProduct` with raster path, label association, identity, GSD, geometry status, and validation status.
2. **Product validation**
   - `samanvaya.validation.benchmark_real.run_real_benchmark`
   - Requires both raster files and validated/partial catalog products. Missing files return `DATA_REQUIRED`.
3. **Raster access**
   - `samanvaya.validation.real_registration._read_primary_band`
   - Opens the supplied raster, checks dimensions, valid values, nodata, and profile.
4. **Metadata-aware configuration**
   - GSD ratio comes from `MissionProduct.gsd_m`.
   - Solar normalization is used only when both products expose sun geometry. Otherwise the run records unavailable illumination metadata.
5. **Registration engine**
   - `lunar_core.pipeline.LunarCorePipeline.register`
   - Existing phase-congruency, dense matcher/RIFT fallback, robust geometry, ANMS, and sub-pixel path are reused.
6. **Registered output**
   - The estimated transform is applied to the original source raster, not only to a normalized display image.
   - `registered_source.tif` is written and reopened immediately.
7. **Match-point product**
   - `matches.csv` and `matches.json` contain coordinates, confidence, inlier state, and sub-pixel availability.
8. **Independent validation**
   - Checkpoints are evaluated separately by the existing transform-based evaluator. Fitting-point reprojection is never promoted to ground-truth accuracy.
9. **Benchmark outputs**
   - `results.json`, `results.csv`, `summary.md`, and per-pair `failure.json` artifacts are written under the requested output directory.

## Evidence boundary

A successful code path or synthetic fixture proves executable infrastructure only. A real scientific claim requires actual mission pixels, a reopened registered product, distributed verified matches, and independent checkpoints.
