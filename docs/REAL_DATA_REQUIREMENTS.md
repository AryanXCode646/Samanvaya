# Real Data Requirements

This contract describes the inputs required before Samanvaya can report real registration or independent accuracy.

## Supported products

| Role | Missions/instruments | Accepted formats | Required state |
|---|---|---|---|
| Source | Chandrayaan-2 OHRC, TMC/TMC-2, IIRS | PDS4 detached raster + label, GeoTIFF where metadata is preserved | `AUTHORIZED`, structured identity, readable raster |
| Reference | LRO NAC, SELENE TC, compatible lunar reference | PDS4 detached raster + label, GeoTIFF | `AUTHORIZED`, structured identity, readable raster |

## Required metadata

- mission and instrument identity from structured metadata or a constrained mission identifier
- product identifier and acquisition timestamp
- raster dimensions and band count
- valid raster association and readable data window
- GSD/pixel-resolution metadata for scale-aware processing
- coordinate/reference-system metadata or an explicit `geometry_status`
- footprint metadata for overlap selection, preferably authoritative product geometry
- nodata/fill and scale/offset when present
- solar geometry when available for illumination diagnostics

IIRS products additionally require:

- authoritative band axis and spatial axes
- band count
- wavelength values, units, and source when available
- valid-data mask and spectral representation configuration

If wavelengths are unavailable, the product must be marked `WAVELENGTH_METADATA_UNAVAILABLE`; no authoritative wavelength claim is allowed.

## Geometry requirements

A geographic corner polygon derived from label latitude/longitude values is an approximate prefilter only. It may produce a `proximity_candidate`, never `CONFIRMED_OVERLAP`. A confirmed pair requires authoritative product geometry or an independently validated projected footprint.

## Validation requirements

A real registration run requires both source and reference rasters to exist. Independent accuracy requires a separate checkpoint file with at least four source/reference coordinates. Reprojection residuals from fitting inliers are not ground-truth accuracy.

## Unsupported or blocked cases

- protected archive products that have not been downloaded by an authorized user
- ambiguous PDS4 label associations
- unknown or conflicting instrument identity
- raster/label dimension disagreement
- unsupported spectral layout or missing required axes
- missing raster files referenced by a benchmark manifest
- products with no usable overlap evidence

These cases must return an explicit non-success status such as `UNVERIFIED`, `INVALID`, `UNSUPPORTED`, `DATA_REQUIRED`, or `BLOCKED_BY_MISSING_DATA`.
