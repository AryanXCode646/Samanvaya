# Scientific Validation Methodology

This repository adopts a conservative, evidence-first validation posture.

## Validation levels

- Level A — software verification: parser, raster, integration, and CLI checks.
- Level B — synthetic algorithm validation: synthetic image pairs with known transform and fixed simulation seed.
- Level C — real mission ingestion validation: metadata and raster access for actual mission products only.
- Level D — real registration / scientific validation: end-to-end registration executed on real mission imagery with independent checkpoints.

## Hard rules

1. Synthetic experiments are never presented as real mission validation.
2. Ground truth must be independent of the fitting points used by the matcher.
3. If a dataset lacks independent checkpoints, the project must not report numeric ground-truth RMSE.
4. Failed scenes are retained in the evidence package and must be summarized in the final report.
5. README claims are generated from the evidence manifest, not manually typed from a narrative guess.

## Evidence boundary in this checkout

The current repository contains metadata validation and fixture-backed parsing checks for representative mission products, but it does not contain a checked-in archive of real mission imagery or an end-to-end real-registration ground-truth dataset. Therefore the evidence-based status remains metadata-and-lazy-access validation only.

## Required pair taxonomy

- OHRC ↔ TMC-2
- OHRC ↔ IIRS
- TMC-2 ↔ IIRS
- Chandrayaan-2 ↔ LRO NAC
- Chandrayaan-2 ↔ SELENE

Each pair must be recorded even when it is Not Run or Not Available. Unknown values are never silently replaced by zero.

## Reproducibility

All scientific claims must be traceable to:

- exact product IDs
- source archive and URL
- local checksums
- Git commit
- frozen configuration hash
- the command used to reproduce the run

This repository stores the current evidence in [evidence/real_data_manifest.json](../evidence/real_data_manifest.json).
