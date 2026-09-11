# Public vs. Local State Audit (Samanvaya SIH PS 26166)

**Audit Timestamp:** 2026-09-11  
**Current Branch:** `main` (synchronized with `hardening-pass`)  
**Commit SHA:** `9532eed`  
**Remote Origin Tracking:** `https://github.com/AryanXCode646/Samanvaya.git`  
**Remote Upstream Tracking:** `https://github.com/ashishsinghbora/Samanvaya.git`  

---

## 1. Executive Summary of Consistency

| Branch / Remote | Current Commit | Status Relative to Public Main | Description |
|---|---|---|---|
| `origin/main` | `9532eed` | Up-to-date | Authoritative public branch; contains full hardened implementation. |
| `local main` | `9532eed` | Up-to-date | Clean working tree; all 167 automated tests passing. |
| `origin/hardening-pass` | `22ceb4b` | Merged into main | Hardening branch fully integrated and pushed. |

**Result:** Zero divergence. The public GitHub repository default landing branch (`main`) now contains the complete, scientifically hardened pipeline, regression tests, claim gates, and experiment protocols.

---

## 2. Component-by-Component Verification

| Component | Status on `main` | Verified Behavior |
|---|:---:|---|
| **Coordinate System Convention** | `VERIFIED` | Strict $T(\text{source FULL\_IMAGE}) = \text{reference FULL\_IMAGE}$ with $(x=\text{col}, y=\text{row})$. No hidden offsets or mixed frames. |
| **Transform Direction & Plausibility** | `VERIFIED` | Forward warping source $\to$ reference; checks positive determinant, condition number, and regional gradients. |
| **Spatial Selection (Match Distribution)** | `VERIFIED` | Primary grid operates strictly in `FULL_SOURCE_IMAGE` coordinates; rebalances source clusters regardless of reference distribution. |
| **Baseline Registration Engine** | `VERIFIED` | Classical SIFT/ORB + RANSAC comparison module (`baseline_registration.py`) enabled for side-by-side benchmarking. |
| **Real Benchmark Manifest & Runner** | `VERIFIED` | Transparent `DATA_REQUIRED` return code when physical flight rasters are absent; exports `scale_results.csv` and `illumination_results.csv`. |
| **Automated Claim Gating** | `VERIFIED` | Strict gating: no synthetic result can declare `PROVEN`; requires independent flight checkpoints for subpixel claims. |
| **Product Inventory (4-Class)** | `VERIFIED` | Distinguishes `AUTHORIZED_REAL`, `SYNTHETIC_FIXTURE`, `UNVERIFIED`, and `INVALID`; only `AUTHORIZED_REAL` allowed in real benchmarks. |
| **IIRS Representation Safeguard** | `VERIFIED` | Multi-band cubes without authoritative wavelength calibration trigger `IIRS_REPRESENTATION_UNCERTAIN`. |
