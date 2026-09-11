# Final Validation Status

Generated for the current repository checkout. This document records evidence boundaries; it does not promote synthetic results to real mission validation.

## Proven

- Full automated suite passes in the current environment.
- Missing real raster inputs are blocked rather than replaced with random arrays.
- Independent checkpoint evaluation applies the executed transform.
- Approximate geographic footprints are not reported as confirmed scientific overlap.
- IIRS representation metadata records wavelength source, units, axis order, and representation configuration.

## Supported

- PDS4/GeoTIFF metadata cataloging for mission products.
- OHRC, TMC-2, IIRS, LRO NAC, and SELENE identity paths where structured metadata supports them.
- Illumination preprocessing, phase-congruency representation, dense matching, robust geometry, sub-pixel refinement, raster output, and match-point exports.
- Local authorized mission-data import and archive reachability status.
- Real registration API and manifest runner are executable, but return `DATA_REQUIRED` when supplied mission pairs or required files are absent.

## Synthetically validated

- Registration, scale-bridge, illumination, IIRS representation, spatial-distribution, and sub-pixel unit/integration tests.
- Bundled benchmark execution and report generation.

## Experimental or partial

- Real IIRS spectral-to-2-D correspondence.
- Authoritative lunar footprint intersection for products with only geographic corner metadata.
- Real cross-modal performance under mission sun-angle, scale, and viewpoint variation.

## Data required

The following cannot be claimed from this checkout because the required authorized products and independent checkpoints are absent:

- Real OHRC ↔ TMC-2 registration validation.
- Real OHRC/TMC-2 ↔ IIRS registration validation.
- Independent real-mission sub-pixel RMSE.
- Real mission spatial-coverage acceptance rates.
- Real LRO/SELENE cross-mission accuracy.

## Reproduction commands

```bash
./.venv/bin/python -m pytest -q
./.venv/bin/python -m compileall -q lunar_core samanvaya tests
./.venv/bin/python -m lunar_core.cli validation --manifest evidence/real_data_manifest.json
./.venv/bin/streamlit run app.py --server.port 8501
```

Real-registration commands:

```bash
./.venv/bin/python -m lunar_core.cli inspect-product /path/to/product.tif
./.venv/bin/python -m lunar_core.cli discover-pairs /path/to/products
./.venv/bin/python -m lunar_core.cli benchmark-real \
	--manifest evidence/real_data_manifest.json \
	--output output/real_benchmark
./.venv/bin/python -m lunar_core.cli verify-output output/real_benchmark/<pair>/registered_source.tif
```

The checked-in evidence manifest currently contains no executable real pair, so `benchmark-real` is expected to exit with `DATA_REQUIRED` and must not create a registered product.

For authorized real data, import products into a manifest, execute the real-pair runner, and provide independent checkpoint files before reporting ground-truth RMSE.
