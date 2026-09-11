# PS 26166 Traceability

This matrix separates implementation from scientific evidence. A feature is not marked validated unless the relevant real products and independent checkpoints exist.

| Requirement | Implementation | Evidence in this checkout | Status |
|---|---|---|---|
| Generic source/reference registration | `lunar_core/pipeline.py`, CLI, Streamlit UI | Synthetic end-to-end tests; real runner accepts supplied products | SUPPORTED; real validation DATA_REQUIRED |
| OHRC/TMC-2 ingestion | Mission catalog, adapters, PDS label parsing | Fixture-backed metadata tests | REAL PRODUCT INGESTION TESTED |
| IIRS ingestion and representations | `lunar_core/preprocessing/spectral.py` | Synthetic spectral tests and explicit provenance model | EXPERIMENTAL; real IIRS DATA_REQUIRED |
| Illumination-aware preprocessing | Photometric normalization and phase congruency | Synthetic illumination tests | SYNTHETICALLY VALIDATED |
| Scale-aware registration | Scale-space and GSD-aware pipeline | Synthetic scale tests | SYNTHETICALLY VALIDATED |
| Correspondence and robust geometry | Dense matcher, RIFT fallback, MAGSAC/USAC paths | Synthetic matcher tests | SUPPORTED; real mission DATA_REQUIRED |
| Uniform spatial distribution | ANMS plus entropy, occupied-grid, extent, edge metrics | Metric unit tests | SUPPORTED; real mission evidence DATA_REQUIRED |
| Sub-pixel refinement | Hessian/Taylor refinement | Synthetic refinement tests | SUPPORTED; real accuracy DATA_REQUIRED |
| Registered product and match points | Raster writer, CSV/JSON/GCP exporters | Synthetic output tests | SUPPORTED; real output DATA_REQUIRED |
| Independent validation | Held-out checkpoint evaluator | Missing-data blocking and transform-based checkpoint tests | SUPPORTED; no real checkpoint set supplied |
| Reproducibility | Commit/config hash in real-run artifacts | Provenance code path | SUPPORTED |
| Physical footprint overlap | Spherical-area helper and explicit approximate prefilter semantics | Geometry and pair-selection tests | PARTIAL; authoritative product geometry DATA_REQUIRED |

## Status vocabulary

- **PROVEN**: independently evidenced on the relevant data.
- **SUPPORTED**: executable code path and tests exist, but mission-grade evidence is absent.
- **EXPERIMENTAL**: implementation exists with synthetic or limited evidence.
- **DATA_REQUIRED**: blocked until authorized mission products/checkpoints are supplied.
- **NOT_SUPPORTED**: intentionally outside the current implementation.

## Current conclusion

Samanvaya is an executable research framework with a conservative real-data path. It is not scientifically validated across real OHRC, TMC-2, and IIRS imagery in this checkout. Synthetic benchmark results must not be presented as spacecraft validation.
