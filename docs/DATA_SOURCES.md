# Authoritative Lunar Mission Data Sources and Specifications (SIH PS 26166)

This document specifies the authoritative data sources, product structures, metadata requirements, and independent validation standards for the Samanvaya lunar registration pipeline.

---

## 1. Supported Planetary Missions & Instruments

| Mission | Instrument | Data Provider / Archive | Primary Product Type | Typical GSD | Spectral / Bands |
|---|---|---|---|---|---|
| **Chandrayaan-2** | **OHRC** (Orbiter High Resolution Camera) | ISRO ISSDC / PRADAN | Calibrated / Raw `.img` + PDS4 `.xml` | $\sim 0.25\text{--}0.32\text{ m/px}$ | 1 band (Panchromatic) |
| **Chandrayaan-2** | **TMC-2** (Terrain Mapping Camera 2) | ISRO ISSDC / PRADAN | Calibrated `.img` / `.tif` + PDS4 `.xml` | $\sim 5.0\text{ m/px}$ | Triplet stereo (Fore, Nadir, Aft) |
| **Chandrayaan-2** | **IIRS** (Imaging Infrared Spectrometer) | ISRO ISSDC / PRADAN | Spectral Cube `.qub` / `.img` + PDS4 `.xml` | $\sim 80.0\text{ m/px}$ | 256 bands ($0.8\text{--}5.0\,\mu\text{m}$) |
| **Lunar Reconnaissance Orbiter (LRO)** | **LROC NAC** (Narrow Angle Camera) | NASA PDS Imaging Node / USGS | Calibrated `.tif` / `.IMG` + PDS label | $\sim 0.50\text{--}1.0\text{ m/px}$ | 1 band (Panchromatic) |
| **SELENE (Kaguya)** | **TC** (Terrain Camera) | JAXA DARTS / SELENE Archive | Calibrated `.img` / `.tif` + label | $\sim 10.0\text{ m/px}$ | Stereo / Panchromatic |

---

## 2. Directory Structure and Data Layout

All local datasets must strictly adhere to the following directory layout:

```text
data/
  missions/
    chandrayaan2/
      ohrc/           # Calibrated OHRC rasters (*.img, *.tif) and PDS4 labels (*.xml)
      tmc/            # Calibrated TMC-2 stereo triplet rasters and PDS4 labels
      iirs/           # IIRS hyperspectral cubes (*.qub, *.img) and PDS4 labels
  reference/
    lro_nac/          # NASA LRO NAC GeoTIFFs (*.tif) and PDS labels
    selene/           # JAXA SELENE TC orthorectified / calibrated rasters
  checkpoints/        # Independent Ground Control Points (*.csv) for held-out evaluation
  manifests/          # Execution manifests conforming to real_benchmark_manifest.schema.json
```

---

## 3. PDS4 Label and Raster Association

1. **Same-Stem Association**:
   Every detached binary raster (e.g. `ch2_ohr_ncp_20211228T2209123959_d_img_d18.img`) must have an accompanying PDS4 XML label with the matching stem (`ch2_ohr_ncp_20211228T2209123959_d_img_d18.xml`) in the same directory.
2. **Explicit Reference Validation**:
   The label must explicitly declare `<file_name>` matching the raster, or share the exact stem. Lone or orphaned XML files without binary rasters are classified as `UNVERIFIED` and cannot be registered.
3. **No Silent Repair**:
   Missing or corrupted metadata (e.g. negative GSD, missing line/sample count) causes the product to be marked `INVALID` or `PARTIAL`. Synthetic defaults are never silently injected into real products.

---

## 4. Metadata Extraction Requirements

For every real mission product, the following metadata fields must be validated:

* **Identity**: `mission`, `instrument`, `spacecraft_name`, `product_id`, `processing_level`.
* **Geometry**: `width` (samples), `height` (lines), `band_count`, `dtype`.
* **Spatial Resolution**: `gsd_m` (Ground Sample Distance in meters/pixel), extracted from label or raster geotransform.
* **Illumination**: `sun_azimuth_deg`, `sun_elevation_deg` (or solar incidence angle), `phase_angle_deg`.
* **Temporal**: `acquisition_time` (UTC ISO 8601).
* **Footprint / Bounding Box**: `center_lat_deg`, `center_lon_deg`, four-corner lat/lon polygon.
* **Integrity**: Streaming SHA-256 hash computed directly from the byte stream on disk.

---

## 5. Independent Checkpoint Contract

To scientifically prove registration accuracy without data leakage, independent checkpoints must be provided:

* **Format**: Comma-separated values (`.csv`).
* **Header**: `point_id,source_x,source_y,reference_x,reference_y,provenance,quality`
* **Coordinate Space**: Strict `FULL_IMAGE` pixel coordinates $(x=\text{column}, y=\text{row})$ on the full uncropped source and reference rasters.
* **Separation of Concerns**:
  * Fit points (correspondences discovered by LoFTR/RIFT) are used **solely** for transformation estimation ($H$).
  * Independent checkpoints are held out and used **solely** for validation scoring.
* **Accuracy Metrics**:
  * $\text{RMSE} = \sqrt{\frac{1}{N}\sum \|T(x_{\text{src}}) - x_{\text{ref}}\|^2}$
  * Evaluated separately for integer predictions vs subpixel continuous predictions.
  * Only real mission data evaluated on independent checkpoints can promote a claim to `PROVEN`.
