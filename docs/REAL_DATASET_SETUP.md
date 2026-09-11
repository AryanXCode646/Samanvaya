# Real Dataset Setup

Samanvaya does not download or fabricate mission data. Place authorized products under this layout:

```text
data/
  missions/chandrayaan2/ohrc/
  missions/chandrayaan2/tmc/
  missions/chandrayaan2/iirs/
  reference/lro_nac/
  reference/selene/
  checkpoints/
  manifests/
```

Each detached PDS4 raster must have an authoritative associated label. A same-stem XML or explicit `file_name` reference is accepted; an unrelated lone XML file is rejected.

Products must expose mission/instrument identity, product ID, dimensions, band count, acquisition time where available, GSD, and geometry status. IIRS additionally needs authoritative axis and wavelength metadata for spectral claims.

Use the inventory command before creating a benchmark manifest:

```bash
./.venv/bin/python -m lunar_core.cli inventory \
  --root data \
  --output evidence/data_inventory.json
```

Discover conservative candidate pairs:

```bash
./.venv/bin/python -m lunar_core.cli discover-real-pairs \
  --root data \
  --output evidence/discovered_real_pairs.json
```

A benchmark manifest must point to actual local rasters and independent checkpoint files. Hashes are recorded in benchmark results; product metadata provenance comes from the PDS4 label or raster metadata. Missing products or checkpoints produce `DATA_REQUIRED` and do not invoke a synthetic fallback.
