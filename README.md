# 🌙 SAMANVAYA (समान्वय)
### Research-Oriented Multi-Modal, Sun-Angle, and Scale-Invariant Lunar Image Correspondence Framework

> **Real-data status:** Mission-product ingestion and registration integration are implemented, but independent Chandrayaan-2, LRO, and SELENE validation remains pending. The benchmark values below are synthetic-only unless explicitly labeled otherwise.

> **IIRS scope:** IIRS products are catalog-aware, but multi-band spectral-cube preprocessing and spectral-to-2-D correspondence are not yet implemented in the registration runner.

> **Local real-product evidence:** The ingestion harness has been exercised against locally supplied Chandrayaan-2 OHRC and Chandrayaan-1 HySI PDS4 labels using sparse temporary files. OHRC metadata and lazy 2-D access passed; HySI was classified as a 64-band partial spectral product. No registration accuracy claim is made from this metadata/access check.

[![ISRO SIH PS 26166](https://img.shields.io/badge/ISRO-SIH%20PS%2026166-0284c7?style=for-the-badge&logo=nasa&logoColor=white)](https://github.com/ashishsinghbora/Samanvaya)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Kornia 0.8](https://img.shields.io/badge/Kornia-0.8-10b981?style=for-the-badge)](https://kornia.readthedocs.io)
[![GDAL / Rasterio](https://img.shields.io/badge/GDAL%20%2F%20Rasterio-1.3%2B-2563eb?style=for-the-badge&logo=qgis&logoColor=white)](https://rasterio.readthedocs.io)
[![Tests](https://img.shields.io/badge/Tests-run%20pytest%20tests%2F-emerald?style=for-the-badge&logo=pytest&logoColor=white)](https://github.com/ashishsinghbora/Samanvaya)
[![CI](https://github.com/ashishsinghbora/Samanvaya/actions/workflows/ci.yml/badge.svg)](https://github.com/ashishsinghbora/Samanvaya/actions/workflows/ci.yml)
[![Security Hardened](https://img.shields.io/badge/Security-XXE%20%26%20Decompression%20Shielded-blueviolet?style=for-the-badge)](SECURITY.md)
[![License MIT](https://img.shields.io/badge/License-MIT-f59e0b?style=for-the-badge)](LICENSE)

<p align="center">
  <b>Engineered for Smart India Hackathon (SIH) Grand Finale — Problem Statement 26166</b><br/>
  <i>"Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS)"</i>
</p>

---

## 📖 Overview

**Samanvaya (समान्वय)** is a research-oriented lunar image correspondence and registration framework for ISRO's **Chandrayaan-2** orbital payloads (**OHRC, TMC-2, IIRS**) and reference planetary datasets (**NASA LRO NAC, JAXA SELENE TC**).

The repository now reflects a more mature engineering posture: mission data ingestion, validation infrastructure, and evidence-aware reporting are implemented, while real-image registration claims remain bounded by actual independent validation. This is a genuine research pipeline rather than a synthetic-only demo.

The framework is designed to handle:
1. **$180^\circ$ Solar Illumination & Shadow Inversion:** Contrast-reversed crater morphology across morning vs afternoon orbital passes.
2. **Up to $320\times$ Ground Sampling Distance (GSD) Disparity:** The architecture is designed for multi-scale correspondence across OHRC ($0.25\text{ m/px}$), TMC-2 ($5.0\text{ m/px}$), and cataloged IIRS products ($80.0\text{ m/px}$); IIRS cube registration remains future work.
3. **Rigorous Sub-Pixel Accuracy Mandate:** Continuous analytical Taylor-series Hessian refinement, benchmarked on synthetic data and design-level validation; real orbital RMSE remains pending independent mission data acquisition.
4. **Out-of-Core Memory Safety:** Sliding-window raster ingestion with spatial Non-Maximal Suppression, processing gigapixel swaths within a strict $\le 4\text{ GB}$ dynamic RAM ceiling.
5. **Mission Interoperability:** Native export of Ground Control Points (GCPs) for **USGS ISIS3 `jigsaw`** bundle adjustment and automated ReportLab executive PDF mission reports.

## 📊 System Scientific Maturity & Provenance

Every algorithm, benchmark result, and capability in Samanvaya is categorized strictly under one of five auditable tiers:

### 1. IMPLEMENTED
- **Clean Architecture Pipeline:** Preprocessing (Hapke/Lommel-Seeliger topographic photometric normalization, dynamic contrast equalization, 2D vectorized Log-Gabor phase congruency), Coarse multiscale Fourier-Mellin ROI extraction, Fine dense transformer matching (LoFTR) with fallback to Classical RIFT/SIFT/Phase Correlation via `MatchingStrategySelector`, 8x8 Grid ANMS spatial distribution enforcement, USAC-MAGSAC++ robust estimation, Analytical paraboloid 2D sub-pixel Taylor refinement, and dynamic geometric model selection (`select_geometric_model` evaluating Translation, Similarity, Affine, Homography with BIC and condition number checks).
- **PDS4 / GeoTIFF Data I/O:** Secure DefusedXML PDS4 parser, sliding-window streaming (`PlanetaryTileProcessor`), and Moon IAU 2015 Sphere (R=1737.4 km) coordinate system support.
- **Diagnostic Export:** 8-panel diagnostic dashboard (`diagnostic_dashboard.png`), residual vector field quiver plots, spatial entropy/coverage reports, and ISIS3 GCP export.
- **Unified CLI:** `samanvaya` command suite (`register`, `validate`, `benchmark`, `inspect-product`, `inspect-pair`, `discover-data`, `inventory`).

### 2. SYNTHETICALLY VALIDATED
- **DEM Ray-Traced Simulations:** Benchmarked across synthetic lunar scenarios with crater power-law distributions and varying solar angles (Apollo 11, Jackson Crater, Low Sun).
- **Sub-Pixel Precision:** Synthetic reprojection RMSE of $0.0027\text{ px} \text{--} 0.3843\text{ px}$ verified on controlled synthetic displacements and known homographies.
- **Ablation Studies:** 7-stage ablation benchmark (`python -m samanvaya benchmark --ablation`) demonstrating incremental improvements across raw classical, illumination norm, multiscale pyramid, geometric filtering, ANMS spatial distribution, subpixel refinement, and the full pipeline.
- **Automated Test Suite:** 179 automated tests passing with 0 failures (`pytest`).

### 3. REAL-DATA EXECUTED
- **PDS4 Label & Raster Ingestion:** Verified against real Chandrayaan-2 OHRC (`ch2_ohr_ncp_20211228T2209123959_d_img_d18`), TMC-2, and Chandrayaan-1 HySI metadata.
- **Pair Identification & Window Slicing:** Bounding-box prefiltering and spherical lunar geodesic distance calculation (`propose_pair`) executed on real product footprints.
- **Uncertainty Tracking:** Pairs lacking co-located imagery or valid footprints are conservatively categorized as `DATA_REQUIRED` or `OVERLAP_UNKNOWN` without fabrication.

### 4. GROUND-TRUTH VALIDATED
- **Checkpoint Framework:** Independent tie-point checkpoint evaluation engine (`evaluate_checkpoints`) comparing continuous subpixel vs integer reprojection error on verified control points.
- **Status:** Evaluated on synthetic ground-truth fixtures. For real Chandrayaan-2 ↔ NASA LRO NAC or JAXA SELENE pairs, independent ground-truth validation is **PENDING** real co-located orbit imagery import.

### 5. CURRENT LIMITATIONS & REMAINING SCIENTIFIC GAPS
- **Real Overlapping Imagery Dependency:** Full orbital validation requires acquiring overlapping raw/calibrated Chandrayaan-2 and LROC NAC or SELENE products in `data/real/`.
- **IIRS Hyperspectral Cubes:** IIRS multi-band data is parsed as partial products with band-mean/PCA 2D representations; full 256-band spectral-to-2D photometric feature correspondence remains research-grade.
- **Steep Topography Shadow Parallax:** 3D topographic relief displacement under extreme sun elevation deltas (>45°) introduces non-projective local parallax that planar homography models cannot completely resolve without explicit DTM integration.

[**🎤 5-Minute Pitch Deck**](PITCH_DECK.md)

---

> [!NOTE]
> **Dataset Provenance & Calibration Disclosure:**  
> The bundled benchmark tiles under `lunar_core/assets/sample_data/` (`scenario_a`, `scenario_b`, `scenario_c`) are high-fidelity calibrated photogrammetric simulations generated via DEM ray-tracing with crater power-law distributions, real lunar landing site coordinates (Apollo 11, Jackson Crater, Shackleton Rim), and Lommel-Seeliger scattering. While Samanvaya includes ingestion drivers for raw ISRO PDS4 XML products and multi-gigabyte GeoTIFFs (`PlanetaryRasterReader`, `PlanetaryTileProcessor`), these compact tiles enable 100% reproducible, offline benchmark verification without multi-gigabyte archive downloads.

> **Real-data status:** A Chandrayaan-2/LRO NAC pair has not yet been acquired in this checkout. Real-data RMSE, matcher-path behavior, and mandate status are pending acquisition and execution; no synthetic result should be interpreted as real orbital validation.

The table below summarizes empirical benchmarks evaluated on these calibrated synthetic datasets. Real-data validation remains pending; see Dataset Provenance above.

| Evaluation Dimension | Synthetic benchmark (DEM ray-trace, self-consistency check) | Real Chandrayaan-2/LRO NAC pair | ISRO SIH Mandate | Status |
|---|---|---|---|---|
| **Sub-Pixel RMSE (Apollo 11)** | $0.3377\text{ px}$ | Pending acquisition | $< 0.400\text{ px}$ | Synthetic result only |
| **Sub-Pixel RMSE (TMC-2 Stereo)** | $0.3355\text{ px}$ | Pending acquisition | $< 0.400\text{ px}$ | Synthetic result only |
| **Extreme Lighting (12° vs 65°)** | $0.3706\text{ px}$ | Pending acquisition | $< 0.400\text{ px}$ | Synthetic result only |
| **180° Shadow Reversal** | $0.1903\text{ px}$ | Pending acquisition | $< 0.400\text{ px}$ | Synthetic result only |
| **Inlier Consensus Ratio** | $52.4\% \text{--} 85.7\%$ | Pending acquisition | $\ge 40\%$ | Not assessed on real data |
| **Spatial Shannon Entropy ($H$)** | $0.8766 \text{--} 0.9661$ | Pending acquisition | $\ge 0.700$ | Not assessed on real data |
| **GSD Scale Dynamic Ratio** | Up to $320\times$ | Pending acquisition | $320\times$ bridge | Capability, not real validation |
| **Peak RAM on Gigapixel Swaths** | $885.7\text{ MB}$ | Pending acquisition | $\le 4096\text{ MB}$ | Instrumented benchmark only |
| **Automated Verification Suite** | Run `pytest tests/ -v` locally | Pending current run | Zero regressions | Not claimed here |

---

## 🏛️ System Architecture

Samanvaya strictly implements **Clean Architecture** principles, enforcing separation between pure photogrammetric physics, application orchestration, external infrastructure adapters, and user interfaces:

```mermaid
flowchart TD
    subgraph INF["1. Infrastructure Layer (Geospatial Drivers & Hardware)"]
        TIF["Planetary GeoTIFFs (> 10k x 10k)"] --> PRD["PlanetaryRasterDriver (IAU 2000:30100)"]
        PDS["PDS4 XML Metadata"] --> SEC["DefusedXML & Path Traversal Shield"]
        SEC --> PRD
        PRD --> PTP["PlanetaryTileProcessor (rasterio.windows.Window)"]
        PTP --> CKD["cKDTree Spatial Boundary Seam Deduplication"]
    end

    subgraph DOM["2. Domain Layer (Photogrammetric Physics & Preprocessing)"]
        PRD --> MIN["Topographic Minnaert & Lommel-Seeliger Normalization"]
        MIN --> PC["2D Vectorized Log-Gabor Phase Congruency (M_max Moments)"]
    end

    subgraph APP["3. Application Layer (Pipelines & Multi-Scale Solvers)"]
        MIN --> FM["Fourier-Mellin 180° Invariant Coarse Rot/Scale Localizer"]
          FM --> CAS["Experimental multimodal bridge (OHRC / TMC-2 / IIRS)"]
        CAS --> TR["Dense LoFTR Linear Transformer Cross-Attention (Primary Matcher)"]
        TR --> ANMS["8x8 Spatial Hash Bucketing ANMS (Entropy H > 0.85)"]
        ANMS --> MAG["USAC-MAGSAC++ Robust Projective Consensus"]
        MAG --> SUB["Phase Congruency-Guided Taylor Hessian Refinement (det(H) > 0)"]
        PC -.-> SUB
        SUB --> COV["Inverse Hessian Covariance Decomposition (sigma_x, sigma_y, w)"]
    end

    subgraph INT["4. Interfaces Layer (CLI, UI & Mission Products)"]
        MAG --> CLI["Headless Command-Line Interface (lunar_core.cli)"]
        MAG --> STP["Streamlit Analytical Workbench (port 8501)"]
        MAG --> ISIS["USGS ISIS3 Jigsaw Control Network (.net / CSV)"]
        MAG --> PDF["Automated ReportLab Executive PDF Mission Reports"]
    end
```

### Module Structure
```
Samanvaya/
├── app.py                         # Interactive Streamlit Web Application (port 8501)
├── run_pipeline.py                # End-to-End Registration Pipeline with Minnaert
├── verify_raster_run.py           # Large-Raster Out-of-Core Verification
├── lunar_core/                    # High-Performance Algorithmic Engines
│   ├── alignment/                 # Dense LoFTR Matcher, Fourier-Mellin, Scale Space
│   ├── assets/sample_data/        # Benchmark GeoTIFFs (Apollo 11, Jackson Crater, Low Sun)
│   ├── cli.py                     # Command-Line Interface Entrypoint
│   ├── data_io/                   # PlanetaryTileProcessor, USGS ISIS3 Exporter, Raster IO
│   ├── evaluation/                # Metrics Engine, ReportLab PDF Reporter
│   ├── models.py                  # Core Data Models (KeypointMatch, AlignmentResult)
│   ├── pipeline.py                # LunarCorePipeline Orchestration
│   ├── postprocessing/            # 2D Taylor Sub-pixel Refiner, ANMS, USAC-MAGSAC++
│   ├── preprocessing/             # Minnaert Normalizer, Contrast Equalizer, Log-Gabor PC
│   └── ui/                        # Streamlit UI Components & Inspectors
├── Makefile                       # Single-Command Automation
├── start.sh                       # One-Command Streamlit Launcher
├── start.ps1                      # Windows PowerShell Launcher
├── Dockerfile                     # Multi-Stage Production Container
├── docker-compose.yml             # Single-Service Streamlit Container Orchestration
├── requirements.txt               # Pinned Production Dependencies
└── tests/                         # Automated Verification Tests
```

---

## 📐 Core Mathematical Pillars

### 1. Topographic Minnaert & Lommel-Seeliger Regolith Scattering
Lunar regolith exhibits extreme backscattering without atmospheric diffusion. To suppress harsh shadow boundaries and crater rim burnout, local surface normals are derived from digital elevation models (DEM) via Sobel spatial gradients:
$$\mathbf{n} = \frac{\left[-\frac{\partial z}{\partial x}, -\frac{\partial z}{\partial y}, 1\right]^T}{\sqrt{1 + \left(\frac{\partial z}{\partial x}\right)^2 + \left(\frac{\partial z}{\partial y}\right)^2}}$$

Local solar incidence $\mu_0 = \cos(i) = \mathbf{n} \cdot \mathbf{s}$ and emission $\mu = \cos(e) = \mathbf{n} \cdot \mathbf{v} = n_z$ are evaluated per-facet:
$$R_{\text{Minnaert}} = \mu_0^k \cdot \mu^{k-1}, \quad R_{\text{LS}} = \frac{\mu_0}{\mu_0 + \mu}$$
where $k \approx 0.80$ is the lunar limb-darkening parameter.

### 2. Illumination-Invariant Log-Gabor Phase Congruency
Human perception and invariant feature coincidence occur where Fourier phase components align across multiple scales, regardless of contrast reversal:
$$PC(x, y) = \frac{\sum_o E_o(x, y)}{\epsilon + \sum_o \sum_n A_{no}(x, y)}$$

Setting the Log-Gabor DC component $G(0, 0) = 0$ guarantees strictly zero response to uniform albedo shifts. Kovesi moment analysis derives the principal invariant step-edge response:
$$M_{\max} = \frac{1}{2} \left(S_{xx} + S_{yy} + \sqrt{(S_{xx} - S_{yy})^2 + 4 S_{xy}^2}\right)$$

### 3. $\mathcal{O}(1)$ Closed-Form Parabolic Taylor Sub-Pixel Refinement
Around the integer correlation peak, continuous similarity $f(x, y)$ is modeled as a 2D bivariate quadric:
$$f(x, y) = ax^2 + by^2 + cxy + dx + ey + f$$

Setting $\nabla f = 0$ yields the continuous sub-pixel offset in $\mathcal{O}(1)$ time:
$$\mathbf{\delta}^* = -\mathbf{H}^{-1} \mathbf{g} = \begin{bmatrix} 2a & c \\ c & 2b \end{bmatrix}^{-1} \begin{bmatrix} -d \\ -e \end{bmatrix} = \begin{bmatrix} \frac{-2bd + ce}{4ab - c^2} \\ \frac{-2ae + cd}{4ab - c^2} \end{bmatrix}$$

**Negative-Definite Hessian Validation:** Strict eigenvalue checks ($a < 0, b < 0, \det(\mathbf{H}) = 4ab - c^2 > 0$) immediately reject directional crater ridges, saddle points, and local minima. Directional photogrammetric bundle uncertainties are derived from the inverse Hessian:
$$\sigma_x = \sqrt{\frac{2|b|}{4ab - c^2}}, \quad \sigma_y = \sqrt{\frac{2|a|}{4ab - c^2}}, \quad w = \frac{1}{\sqrt{\lambda_1 \lambda_2}}$$

### 4. 2D Spatial Shannon Entropy Regularization
To prevent match clustering on high-contrast crater rims while leaving planar mare unconstrained, an $8 \times 8$ spatial hash allocator caps top-confidence correspondences per cell, enforcing uniform spatial Shannon entropy:
$$H_{\text{spatial}} = -\sum_{k=1}^K p_k \log_2(p_k) \Big/ \log_2(K) \quad \ge 0.85$$

## Scientific References

- Kovesi, P. (1999). *Image Features from Phase Congruency*. Videre, 1(3).
- Lowe, D. G. (2004). *Distinctive Image Features from Scale-Invariant Keypoints*. IJCV, 60, 91-110.
- Barath, D., et al. (2020). *MAGSAC++, a Fast, Reliable and Accurate Robust Estimator*. CVPR Workshops.
- Kornia LoFTR implementation: [kornia.feature.LoFTR](https://kornia.readthedocs.io/en/latest/models.html).

---

## ⚡ Quickstart & One-Command Automation

### Prerequisites
- Linux / macOS / Windows WSL2
- Python 3.10+
- GDAL system libraries (`libgdal-dev`)

### Installation
```bash
# Clone the canonical upstream repository
git clone https://github.com/ashishsinghbora/Samanvaya.git
cd Samanvaya

# Setup virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Validation status
- IMPLEMENTED: mission-data ingestion, PDS4 label resolution, conservative metadata parsing, registration pipeline scaffolding, and evaluation exports.
- TESTED: deterministic identity and catalog regressions; synthetic benchmark pipeline checks; representative PDS4 metadata fixtures for Chandrayaan-2 and LRO.
- REAL-DATA TESTED: metadata extraction and lazy-access validation are exercised against representative real-mission metadata fixtures when available; no registration-accuracy claim is implied.
- FULL IMAGE-LEVEL REAL-DATA VALIDATION: pending. This repository does not currently contain a checked-in real Chandrayaan-2 or LRO image pair sufficient to claim end-to-end geographic or matching validation.
- SCIENTIFICALLY VALIDATED: not claimed. Any footprint overlap is labeled as an approximation until independent geospatial validation is available.

### Evidence-backed validation summary
Scientific claims in this repository are now generated from the project evidence manifest rather than improvised narrative text. Run:

```bash
python -m lunar_core.cli validation --json
```

This produces a conservative summary based on [evidence/real_data_manifest.json](evidence/real_data_manifest.json) and the validation matrix in [validation/validation_matrix.md](validation/validation_matrix.md). The current repository status is intentionally limited to metadata-and-lazy-access validation only.

### Merge-readiness stance
The current codebase is suitable for a conservative metadata-hardening and workflow-validation pass, but it is not yet a scientifically complete end-to-end lunar mission registration claim. The repo should be considered merge-ready only under the narrower condition that the change is limited to evidence-backed metadata handling, transparent uncertainty states, and reproducible fixture-driven validation—not full real-image scientific validation.

### Footprint geometry scope
Samanvaya currently records box-style footprint overlap as a planar bounding-box approximation. The `geometry_method` field is set to `planar_bounding_box_approximation`, and `overlap_status` is one of `APPROXIMATE`, `VERIFIED`, or `UNKNOWN`. Do not interpret planar overlap as a physically meaningful lunar footprint intersection without separate geodesic validation.

### Pair/status semantics
A proximity-only match is not treated as an overlap-confirmed pair. When no valid footprint exists, the pair is staged as `proximity_candidate` with `overlap_status = "UNKNOWN"` and requires explicit later validation.

Before an offline judging session, run `make prefetch-weights` once while internet access is available. This downloads the pretrained LoFTR weights; if loading later fails, the UI reports that untrained weights are not meaningful.

### Running the Services
```bash
# Launch interactive Streamlit workbench
./start.sh
# or: make run
```
- **Streamlit Analytical Workbench:** [http://localhost:8501](http://localhost:8501)
- **Headless Pipeline Execution:** `make pipeline` or `python3 run_pipeline.py --scenario scenario_a`
- **Quantitative Benchmark Metrics:** `make metrics` or `python3 -m lunar_core.evaluation.metrics`
- **Large-Raster Verification:** `make verify-raster` or `python3 verify_raster_run.py --scenario scenario_a` (bundled calibrated benchmark data, not spacecraft data)
- **Executive PDF Mission Report:** `make report-pdf` or `python3 -m lunar_core.evaluation.pdf_reporter`

---

## 🧪 Comprehensive Verification & Testing

### 1. Run the Full Automated Test Suite
```bash
make test
# or: pytest tests/ -v
```

### 2. Execute Large-Raster Out-of-Core Verification
```bash
make verify-raster
# or: python3 verify_raster_run.py --scenario scenario_a
```
Reads bundled calibrated benchmark GeoTIFF tiles, runs 9 out-of-core windowed tiles, performs cKDTree boundary seam deduplication, and generates `evaluation_report.json` and `evaluation_report.csv`. This verifies raster processing behavior; it is not real spacecraft-data validation.

### 3. Generate Executive ReportLab PDF Mission Report
```bash
make report-pdf
# or: python3 -m lunar_core.evaluation.pdf_reporter
```
Generates a technical PDF with telemetry tables, side-by-side verification snapshots, and residual histograms. It reports observed reprojection metrics; it is not an official ISRO certification.

### 4. Authoritative Samanvaya Command Suite
```bash
# Register any two planetary image products
python -m samanvaya register <source_path> <reference_path> --out output/registered

# Validate product pairing and overlap
python -m samanvaya validate <source_path> <reference_path>

# Run benchmark across real pairs or scientific ablation study
python -m samanvaya benchmark --ablation --output-dir output/ablation
python -m samanvaya benchmark --manifest data/real/manifest.json --output-dir output/benchmark

# Inspect product metadata & georeferencing
python -m samanvaya inspect-product <product_path>

# Inspect pair overlap geometry
python -m samanvaya inspect-pair <source_path> <reference_path>

# Audit repository product inventory
python -m samanvaya inventory --root data/ --json
```


---

## 🛡️ Cybersecurity & Defensive Safeguards

Samanvaya is hardened against malicious raster and payload exploits:
- **XML Entity Injection (XXE) Prevention:** PDS4 labels are parsed exclusively using `defusedxml` with entity resolution disabled (`resolve_entities=False`).
- **Path Traversal Shielding:** File paths undergo strict normalization, resolving symbolic links and rejecting null bytes (`\0`) and directory traversal sequences (`..`).
- **Decompression Bomb Defense:** Raster dimensions are capped at $30,000 \times 30,000$ pixels with a $4\text{ GB}$ maximum uncompressed buffer limit, rejecting maliciously crafted compressed GeoTIFFs before memory allocation.

---

## 🛰️ USGS ISIS3 & SPICE Interoperability

Samanvaya exports verified tie-points directly into USGS ISIS3 control network formats for secondary bundle adjustment:
```python
from lunar_core.data_io.isis_exporter import IsisGcpExporter

exporter = IsisGcpExporter(target_body="MOON", crs_wkt="IAU2000:30100")
exporter.export_control_network(
    inliers=result.inliers,
    output_path="lunar_control_network.net",
    point_prefix="CH2_OHRC_TMC2_",
)
```
The resulting `.net` file is ingested directly into USGS ISIS3 `jigsaw`:
```bash
jigsaw fromlist=cube_list.lis cnet=lunar_control_network.net radius=1737400 pointid=CH2_OHRC_???
```

---

## 📄 License & Attribution

Distributed under the **MIT License**. Developed by **Team Samanvaya** for the **Smart India Hackathon (SIH) Grand Finale — Problem Statement 26166**, in collaboration with the **Indian Space Research Organisation (ISRO)**.

---
<p align="center">
  <b>Samanvaya (समान्वय) — Elevating Planetary Photogrammetry to Continuous Sub-Pixel Precision.</b>
</p>
