# Mission Data

Place downloaded mission products under `data/raw/` and keep the original
image files beside their labels. This directory is ignored so mission data
and credentials are never committed.

Scan products without loading raster pixels:

```bash
python -m lunar_core.data_io.mission_catalog data/raw/ --output data/metadata/products.csv
```

The catalog records validation failures instead of silently discarding them.
Real-data validation requires locally supplied products and is distinct from
the repository's synthetic regression benchmark.