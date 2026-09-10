# Mission Data

Place downloaded mission products under `data/raw/` and keep the original
image files beside their labels. This directory is ignored so mission data
and credentials are never committed. Supported local products include detached
PDS4 `.img` images and `.qub` spectral cubes; large files are accessed lazily.

Suggested layout:

```text
data/raw/chandrayaan2/ohrc/
data/raw/chandrayaan2/tmc2/
data/raw/chandrayaan2/iirs/
data/raw/lro/nac/
data/raw/selene/
data/metadata/
data/processed/
```

Official acquisition portals:

- Chandrayaan-2: https://chmapbrowse.issdc.gov.in/
- LRO NAC: https://lroc.im-ldi.com/data/
- SELENE/KAGUYA: https://darts.isas.jaxa.jp/app/pdap/selene/

Do not bypass authentication controls or commit downloaded mission data.

Scan products without loading raster pixels:

```bash
python -m lunar_core.data_io.mission_catalog data/raw/ --output data/metadata/products.csv
```

The catalog records validation failures instead of silently discarding them.
Real-data validation requires locally supplied products and is distinct from
the repository's synthetic regression benchmark.