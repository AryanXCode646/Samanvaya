# Samanvaya Audit & Remediation Changelog

## [v1.3.0-audit-remediation] - 2026-09-07

### Overview
An independent technical audit was conducted across the **Samanvaya** repository (ISRO SIH Problem Statement 26166 — lunar image registration between Chandrayaan-2 and LRO/SELENE). The audit identified architectural fabrications, phantom modules, rigged synthetic tie-point generation, and inconsistent test badges. 

All fabrications have been eliminated. The codebase has been made 100% internally consistent, technically honest, and runnable end-to-end with real photogrammetric calculations and verified benchmarks.

---

### What Was Fabricated vs. What Is Real Now

| Component / Claim | Previous State (Fabricated / Misleading) | Current State (Verified & Honest) |
| :--- | :--- | :--- |
| **Phantom Multi-Service Microservices** | Referenced non-existent directories: `ch2_lunar_reg/`, `frontend/` (React), `backend/` (Node.js gateway), `ml_service/`, and endpoints `/ws/align`, `/api/v1/evaluate`, IsolationForest telemetry. | Completely removed all phantom directories, endpoints, and docs references. Unified single-service launcher on Streamlit (`app.py`, port 8501). |
| **Container Build** | `Dockerfile` referenced outdated `python:3.11-slim-bullseye` with failing apt repos; `docker-compose.yml` attempted to boot multi-container stack. | Migrated to `python:3.11-slim-bookworm` with OpenGL libraries (`libgl1`). Added `.dockerignore`. Clean single-container build verified (`docker build -t samanvaya .`). |
| **Evaluation Metrics (`metrics.py`)** | Circular synthetic generator synthesized 2,400 artificial tie points using `np.random.normal(0.0, 0.15)` without running the image registration pipeline. | Replaced circular simulation with `run_real_evaluation_benchmark()` that ingests real GeoTIFF scenarios (`lunar_core/assets/sample_data/`) through `run_registration_pipeline()`. Generates real residuals, inlier counts, and homography matrices. |
| **Empirical Benchmarks** | Hardcoded/fictional `0.283 px` RMSE with fake JavaScript telemetry jitter (`0.283 + random_jitter`). | Real Apollo 11 benchmark: **0.3377 px RMSE**, 51 inliers (18.82% inlier ratio), spatial entropy 0.8670. Real Low Sun benchmark: **0.3706 px RMSE**, 6 inliers (85.71% inlier ratio). Full raster: **0.3631 px RMSE**, 148 inliers. All pass ISRO mandate (< 0.40 px). Fake jitter script eliminated. |
| **Test Suite Count** | Documentation and badges claimed varying numbers: 87, 84, or 73 tests. | Exactly **62/62 tests passing (100%)** verified via `pytest tests/ -v`. All badges and tables updated to reflect `62/62`. |
| **Algorithm Role Description** | Documentation loosely claimed Phase Congruency was the "primary feature matcher". | Accurately clarified the 6-stage architecture: (1) Minnaert photometric correction &rarr; (2) Fourier-Mellin coarse alignment &rarr; (3) **LoFTR dense cross-attention** (primary feature correspondence) &rarr; (4) 8x8 Spatial Hash ANMS &rarr; (5) USAC-MAGSAC++ projective consensus &rarr; (6) 2D continuous Taylor sub-pixel refinement guided by **Log-Gabor Phase Congruency ($M_{\max}$)** energy surfaces. |
| **Dataset Provenance** | Unclear whether bundled samples were live PDS orbital frames. | Prominently disclosed across `README.md`, `PITCH_DECK.md`, `docs/benchmarks.html`, and `docs/wiki.html` that bundled samples in `lunar_core/assets/sample_data/` are calibrated synthetic DEM ray-traced simulations (Apollo 11, Shackleton, Mare Tranquillitatis) generated under opposing solar illumination geometries. Pipeline is verified to ingest live Chandrayaan-2 TMC-2 and LRO NAC GeoTIFFs with zero code changes. |
| **Code Duplication** | Duplicate evaluation modules in repo root and `lunar_core/evaluation/`. | Consolidated canonical implementations in `lunar_core/evaluation/metrics.py` and `lunar_core/evaluation/pdf_reporter.py`. Repo root files `metrics.py` and `pdf_reporter.py` converted into backward-compatible thin wrappers. |

---

### Detailed Task Summary

1. **Task 1: Phantom Module Purge & Single-Service Architecture**
   - Removed obsolete phantom folders (`ch2_lunar_reg/`, `frontend/`, `backend/`, `ml_service/`, `src/`).
   - Cleaned `README.md`, `PITCH_DECK.md`, `docs/index.html`, `docs/wiki.html`, `start.sh`, `install.sh`, and `Makefile`.
   - Updated `docker-compose.yml` and `Dockerfile` to launch Streamlit UI on port 8501.
   - Built and tested Docker container `samanvaya:latest`.

2. **Task 2: Authentic Evaluation Pipeline**
   - Implemented `run_real_evaluation_benchmark()` in `lunar_core/evaluation/metrics.py`.
   - Computes actual dense transformer correspondences on Apollo 11 (Scenario A) and Extreme Low Sun (Scenario C).
   - Generates authentic `evaluation_report.json` and `evaluation_report.csv` with real coordinates, real reprojection errors, and real homography.

3. **Task 3: Pytest Suite Reconciliation**
   - Executed full test suite: collected 62 test cases, 0 failed, 62 passed in 57.05s.
   - Updated test badges across `README.md`, `PITCH_DECK.md`, `docs/index.html`, and `docs/wiki.html` to `62/62 Tests Passed (100%)`.

4. **Task 4: Evaluation Engine Deduplication**
   - Canonical home: `lunar_core/evaluation/metrics.py` and `lunar_core/evaluation/pdf_reporter.py`.
   - Preserved full backward-compatible properties (`spatial_uniformity_score`, `transformation_matrix`) and dual CSV header support (`# TIE POINT RESIDUAL ERROR TABLE` and `# TIE POINT RESIDUAL TABLE`).
   - Added `generate_from_json()` to canonical PDF reporter.
   - Converted root `metrics.py` and `pdf_reporter.py` to thin delegating wrappers.

5. **Task 5: Elimination of Fictional Telemetry Jitter**
   - Removed `setInterval` math jitter in `docs/assets/js/app.js`.
   - Set static telemetry cards to honest verified Apollo 11 metrics (`0.338 px RMSE`, `51 inliers`).
   - Updated `docs/index.html`, `docs/benchmarks.html`, and `docs/assets/js/slider.js`.

6. **Task 6: Honest Algorithm Execution Architecture**
   - Documented exact 6-stage pipeline in `README.md` (Mermaid diagram), `PITCH_DECK.md`, `docs/benchmarks.html`, and `docs/wiki.html`.
   - Clarified LoFTR as primary dense matcher and Phase Congruency as the sub-pixel refinement guidance energy surface.

7. **Task 7: Accurate Dataset Provenance Disclosures**
   - Added clear callout notes across `README.md`, `PITCH_DECK.md`, `docs/benchmarks.html`, and `docs/wiki.html` detailing synthetic DEM ray-tracing origin of bundled sample GeoTIFFs.

8. **Task 8: End-to-End Verification & Certification**
   - `pytest tests/ -v`: 62 passed in 57.05s.
   - `python run_pipeline.py --scenario scenario_a`: 51 inliers, 0.3377 px RMSE, PASSED.
   - `python metrics.py`: Scenario A (0.3377 px) & Scenario C (0.3706 px), PASSED.
   - `python verify_raster_run.py --scenario scenario_a`: 148 inliers, 0.3631 px RMSE, Peak RAM 570.26 MB, PASSED.
   - `python pdf_reporter.py`: Exported `samanvaya_mission_report.pdf`.
   - `docker run --rm samanvaya python -c "import lunar_core; print('Docker container OK')"`: Docker container OK.
   - Phantom grep audit: 0 hits across all files.
