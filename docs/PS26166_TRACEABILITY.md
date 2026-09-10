# PS 26166 Traceability Matrix

This document is the evidence-first audit for the Samanvaya project against the SIH problem statement for PS 26166.

Scope note:
- This matrix distinguishes engineering implementation from scientifically validated production evidence.
- A requirement is marked complete only when the implementation exists, the production code path uses it, tests verify it, and real-data evidence exists or is explicitly marked unavailable.
- Where real data cannot be supplied in this checkout, the status is marked as `NOT VALIDATED`, `PARTIAL`, or `BLOCKED BY DATA AVAILABILITY`, not as complete.

## Traceability Matrix

| Requirement | Implementation | Files | Tests | Real-data Evidence | Status | Remaining Gap |
|---|---|---|---|---|---|---|
| 1. Generic software solution | Registration pipeline and CLI/UI orchestrate a reusable multi-stage flow for image alignment and evaluation. | `lunar_core/pipeline.py`, `lunar_core/cli.py`, `lunar_core/ui/app.py` | `tests/test_lunar_core.py`, `tests/test_cli_workflows.py`, `tests/test_ui_app.py` | Synthetic benchmark scenes and metadata-only real-product scans; no full real pair execution proven. | PARTIAL | Need a fully generic `MissionProduct`/`ProductPair` path that is exercised with real mission products instead of synthetic scenes. |
| 2. OHRC support | Mission metadata and raster-aware support exist for OHRC. | `lunar_core/data_io/mission_catalog.py`, `lunar_core/data_io/raster_reader.py`, `lunar_core/models.py` | `tests/test_mission_catalog.py`, `tests/test_real_mission_fixtures.py` | Metadata fixtures exist; real OHRC image validation not yet executed end-to-end. | PARTIAL | Real OHRC registration evidence is still missing. |
| 3. TMC/TMC-2 support | TMC/TMC-2 identity and catalog awareness are implemented. | `lunar_core/data_io/mission_adapters.py`, `lunar_core/data_io/product_identity.py`, `lunar_core/models.py` | `tests/test_mission_catalog.py`, `tests/test_label_identity.py` | Metadata fixtures and catalog parsing exist; real TMC/TMC-2 image registration not executed. | PARTIAL | Need a real TMC/TMC-2 pair through full production registration. |
| 4. IIRS support | Spectral cube and metadata handling exists as a spectral foundation. | `lunar_core/preprocessing/spectral.py`, `lunar_core/alignment/scale_space.py`, `lunar_core/data_io/mission_catalog.py` | `tests/test_iirs_alignment.py` | Synthetic IIRS cube tests exist; no real IIRS mission product proved in full registration flow. | PARTIAL | Need real IIRS spectral metadata and an actual registration representation from real cube data. |
| 5. Lunar reference imagery | LRO/SELENE reference mission metadata and pair logic exist. | `lunar_core/data_io/mission_adapters.py`, `lunar_core/data_io/pair_selector.py`, `README.md` | `tests/test_mission_catalog.py`, `tests/test_real_mission_fixtures.py` | Real metadata fixtures exist; no real LRO/SELENE pair validation executed. | PARTIAL | Need actual LRO and SELENE reference pair execution and overlap validation. |
| 6. Illumination variation | Photometric normalization, phase congruency, and Fourier/Mellin-style invariances are implemented. | `lunar_core/preprocessing/photometric.py`, `lunar_core/preprocessing/phase_congruency.py`, `lunar_core/alignment/fourier_mellin.py` | `tests/test_phase_congruency_visual.py`, `tests/test_photometric_dem.py` | Synthetic evaluation only. | PARTIAL | Demonstrate robustness under real lighting variation using actual mission metadata and held-out checks. |
| 7. Viewpoint variation | Geometric registration model supports affine/projective transforms. | `lunar_core/alignment/*.py`, `lunar_core/pipeline.py`, `lunar_core/evaluation/metrics.py` | `tests/test_lunar_core.py`, `tests/test_dense_loftr_matcher.py` | Synthetic success only; no independent real-data viewpoint test. | PARTIAL | Need real cross-mission viewpoint evidence with independent checkpoints. |
| 8. Scale variation | GSD-aware and multi-scale logic exists and is represented in the pipeline architecture. | `lunar_core/alignment/scale_space.py`, `lunar_core/data_io/raster_reader.py`, `lunar_core/pipeline.py` | `tests/test_iirs_alignment.py`, `tests/test_dense_loftr_matcher.py` | Mostly synthetic; no verified real scan with actual GSD mismatch. | PARTIAL | Need real mission-pair scale tests with measured GSD ratios and actual overlap evidence. |
| 9. Corresponding match points | Dense and RIFT matching produce candidate correspondences; match outputs are generated in evaluation/reporting. | `lunar_core/alignment/dense_matcher.py`, `lunar_core/alignment/rift_matcher.py`, `lunar_core/ui/app.py` | `tests/test_dense_loftr_matcher.py`, `tests/test_fallback_matching.py` | Synthetic data; no mission-pair correspondence outputs from real mission data. | PARTIAL | Need output of actual corresponding points from real mission products, with provenance. |
| 10. Registered product | The pipeline can warp and visualize source to reference frame. | `lunar_core/pipeline.py`, `lunar_core/ui/app.py`, `lunar_core/data_io/raster_writer.py` | `tests/test_run_pipeline.py`, `tests/test_ui_app.py` | Registration output is validated on synthetic benchmark data. | PARTIAL | Need a real registered product from a real mission pair with traceable metadata and output. |
| 11. Sub-pixel refinement | Analytical sub-pixel refinement exists. | `lunar_core/postprocessing/subpixel.py`, `lunar_core/postprocessing/anms.py` | `tests/test_subpixel.py` | Synthetic confidence only; not independently validated on real reference points. | PARTIAL | Need independent checkpoint evaluation on real mission products. |
| 12. Uniform match distribution | Spatial entropy and distribution heuristics exist. | `lunar_core/evaluation/metrics.py`, `lunar_core/postprocessing/anms.py` | `tests/test_evaluation_metrics.py`, `tests/test_lunar_core.py` | Synthetic-only evaluation. | PARTIAL | Need real mission-pair spatial coverage metrics and explicit failure analysis when clustering occurs. |
| 13. RMSE | RMSE metrics are computed and exported. | `lunar_core/evaluation/metrics.py`, `lunar_core/models.py` | `tests/test_evaluation_metrics.py`, `tests/test_lunar_core.py` | Synthetic and reprojection-only metrics exist; real independent ground-truth RMSE is not demonstrated. | PARTIAL | Must separate reprojection RMSE from ground-truth RMSE and evaluate using held-out checkpoints. |
| 14. Inlier count | Inlier count is part of the report and model. | `lunar_core/evaluation/metrics.py`, `lunar_core/models.py` | `tests/test_evaluation_metrics.py`, `tests/test_lunar_core.py` | Synthetic/inlier count is present, but not tied to independent real-data validation. | PARTIAL | Need real inlier counts from real mission pair validation. |
| 15. Inlier ratio | Inlier ratio is computed for evaluation. | `lunar_core/evaluation/metrics.py`, `lunar_core/models.py` | `tests/test_evaluation_metrics.py`, `tests/test_lunar_core.py` | Synthetic/inlier-ratio benchmarks exist; not mission-validated. | PARTIAL | Need real mission-pair inlier ratio and failure reporting. |
| 16. Real-data validation | The project contains metadata and lazy-access checks for representative local real-mission fixtures, plus a validation framework and evidence manifests. | `evidence/real_data_manifest.json`, `lunar_core/validation.py`, `validation/validation_matrix.md`, `README.md` | `tests/test_validation_framework.py` | Validated only at the metadata/lazy-access level. End-to-end real registration not executed. | PARTIAL | Need actual registration execution on local authorized real data and, if unavailable, a clear `BLOCKED BY DATA AVAILABILITY` status. |
| 17. Reproducibility | CLI and validation harness provide reproducible commands and summary outputs. | `lunar_core/cli.py`, `validation/README.md`, `validation/summary.json` (if present), `README.md` | `tests/test_cli_workflows.py`, `tests/test_validation_framework.py` | Reproducibility is structured for synthetic and metadata workflows, but not for mission-grade end-to-end real-data runs. | PARTIAL | Need real experiment provenance, checksums, config hashes, and exact CLI reproduction steps for real mission pairs. |

## Consolidated Assessment

The project has reached a solid engineering baseline for:
- generic alignment orchestration,
- metadata-driven mission/product discovery,
- synthetic evaluation routines,
- conservative validation summaries,
- and evidence documentation.

However, the project remains scientifically incomplete with respect to the actual SIH requirement because the following are not yet demonstrated with real mission data and independent validation:

1. real OHRC registration on actual data,
2. real TMC/TMC-2 registration on actual data,
3. real IIRS spectral-to-2D correspondence under actual cube metadata,
4. real LRO/SELENE reference registration, and
5. independent checkpoint validation that separates fitting points from held-out validation points.

## Conclusion

The repo is not scientifically complete for PS 26166 as written. It is a solid research and engineering foundation with a truthful evidence boundary, but it should not be described as fully validated until the real-data path is executed and independently evidenced.
