# Real Dataset Setup & Execution Guide

Samanvaya adheres to a strict scientific rule: **we never fabricate mission imagery, control points, or validation scores**.

When real mission data is not yet downloaded to the local filesystem, Samanvaya transparently reports `DATA_REQUIRED` and refuses to produce fake registration outputs.

---

## 1. Directory Structure

Place all downloaded mission data in the authoritative folder hierarchy:

```text
data/
  missions/
    chandrayaan2/
      ohrc/           # Calibrated OHRC rasters (*.img, *.tif) + PDS4 labels (*.xml)
      tmc/            # Calibrated TMC-2 stereo triplet rasters + PDS4 labels
      iirs/           # IIRS hyperspectral cubes (*.qub, *.img) + PDS4 labels
  reference/
    lro_nac/          # NASA LRO NAC GeoTIFFs (*.tif) + PDS labels
    selene/           # JAXA SELENE TC calibrated rasters
  checkpoints/        # Independent Ground Control Points (*.csv)
  manifests/          # Benchmark manifests (*.json)
```

---

## 2. Ingestion & Inventory Verification

Before creating a benchmark manifest, verify local products using the inventory command:

```bash
# Via Samanvaya CLI
./.venv/bin/python -m samanvaya inventory \
  --root data \
  --output evidence/data_inventory.json

# Or via lunar_core CLI
./.venv/bin/python -m lunar_core.cli inventory \
  --root data \
  --output evidence/data_inventory.json
```

The inventory scanner inspects every raster header and PDS4 label, classifying each product into:
1. `AUTHORIZED_REAL`: Real flight data with validated metadata, real raster on disk, recognized mission.
2. `SYNTHETIC_FIXTURE`: Located in synthetic fixture directories or generated for deterministic unit tests.
3. `UNVERIFIED`: Real file present on disk, but missing PDS4 label, incomplete metadata, or partial status.
4. `INVALID`: Parsing failed, corrupted file, zero dimensions, or unsupported data format.

**Only `AUTHORIZED_REAL` products are permitted to enter the real scientific benchmark.**

---

## 3. Candidate Pair Discovery

Discover overlapping candidate pairs between source and reference products:

```bash
./.venv/bin/python -m lunar_core.cli discover-real-pairs \
  --root data \
  --output evidence/discovered_real_pairs.json
```

---

## 4. Benchmark Execution

Run the real benchmark runner against a populated manifest:

```bash
./.venv/bin/python -m lunar_core.cli benchmark-real \
  --manifest evidence/real_data_manifest.json \
  --output output/real
```

### Expected Behavior
* **When physical mission rasters are absent**: Exits cleanly with code `2` and status `DATA_REQUIRED`. Generates `results.json`, `scale_results.csv`, and `illumination_results.csv` recording missing files.
* **When valid mission rasters are present**: Executes full registration pipeline, outputs `registered_source.tif`, verifies output reopening, computes independent held-out metrics, and generates claim gates.
