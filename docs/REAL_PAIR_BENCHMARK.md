# Real Pair Benchmark Report Template

This document is intentionally populated only from actual imported mission files and measured outputs. It must never include invented values.

## Pair Summary

- Pair ID: {PAIR_ID}
- Mission pair: Chandrayaan-2 OHRC ↔ LROC NAC
- Source product ID: {SOURCE_PRODUCT_ID}
- Reference product ID: {REFERENCE_PRODUCT_ID}
- Source mission: {SOURCE_MISSION}
- Source instrument: {SOURCE_INSTRUMENT}
- Reference mission: {REFERENCE_MISSION}
- Reference instrument: {REFERENCE_INSTRUMENT}
- Acquisition time: {ACQUISITION_TIME}
- Source resolution (m/px): {SOURCE_GSD_M}
- Reference resolution (m/px): {REFERENCE_GSD_M}
- Source dimensions: {SOURCE_WIDTH} x {SOURCE_HEIGHT}
- Reference dimensions: {REFERENCE_WIDTH} x {REFERENCE_HEIGHT}
- Overlap: {OVERLAP_STATUS}
- Match count: {MATCH_COUNT}
- Inlier count: {INLIER_COUNT}
- Inlier ratio: {INLIER_RATIO}
- Estimated transformation: {ESTIMATED_TRANSFORMATION}
- Runtime: {RUNTIME_MS} ms
- Memory: {MEMORY_MB} MB
- Ground-truth status: {GROUND_TRUTH_STATUS}
- RMSE against ground truth: {RMSE_PX}
- P95 against ground truth: {P95_PX}
- Absolute accuracy: {ABSOLUTE_ACCURACY}

## Provenance

- Source checksum SHA256: {SOURCE_SHA256}
- Reference checksum SHA256: {REFERENCE_SHA256}
- Source path: {SOURCE_PATH}
- Reference path: {REFERENCE_PATH}
- Artifact directory: {ARTIFACT_DIR}

## Scientific Status

- Independent ground truth required: YES
- RMSE against ground truth = PENDING until independent control points are supplied.
- P95 against ground truth = PENDING until independent control points are supplied.
- Absolute registration accuracy = PENDING until independent control points are supplied.
