# Public vs. Local State Audit (Samanvaya SIH PS 26166)

**Audit Timestamp:** 2026-09-11  
**Current Local Branch:** `hardening-pass`  
**Current Commit:** `e7a7310`  
**Remote Origin Tracking:** `https://github.com/AryanXCode646/Samanvaya.git`  
**Remote Upstream Tracking:** `https://github.com/ashishsinghbora/Samanvaya.git`  

---

## 1. Executive Summary of Divergence

| Branch / Remote | Latest Commit | Status Relative to Public Main | Description |
|---|---|---|---|
| `origin/main` | `0d271b3` | Baseline merged | Contains PR #24 merge (`c4f7521`). |
| `origin/hardening-pass` | `eb99db9` | +2 commits ahead of PR #24 | Pushed during earlier hardening pass (full-image geometry). |
| `local hardening-pass` | `e7a7310` | +3 commits ahead of PR #24 | Contains latest baseline comparison, real registration workflow, and experiment protocol. |

The local `hardening-pass` branch contains critical scientific fixes, coordinate frame audits, baseline comparisons, and experiment protocols that are not yet in `origin/main`.

---

## 2. Component-by-Component Comparison

| Component | Local State (`HEAD`) | Public State (`origin/main`) | Difference | Required Action |
|---|---|---|---|---|
| **ROI / Target Matcher Coordinate Frame** | Aligned common grid coordinates correctly separated from reference ROI offsets (`samanvaya/registration/coordinates.py`). | Fixed in PR #24 merge; verified locally. | Fully consistent; coordinate frame verified. | Preserve and guard with coordinate propagation regression tests. |
| **Transform Direction Convention** | Strict canonical $T(\text{source FULL\_IMAGE}) = \text{reference FULL\_IMAGE}$ with $(x=\text{col}, y=\text{row})$. | PR #24 established transform convention. | Fully consistent. | Maintain explicit frame assertions. |
| **Spatial Selection (Match Distribution)** | Operates in `FULL_SOURCE_IMAGE` coordinates; enforces source-space uniform distribution. | Partial (previously applied grid capping on reference frame). | Local hardening enforces source-image primary spatial grid. | Enforce source-space capping across LoFTR, ANMS, and metrics. |
| **Baseline Registration Engine** | `samanvaya/validation/baseline_registration.py` (Classical SIFT/ORB + RANSAC comparison). | Absent on `origin/main`. | Local module enables direct comparative baseline benchmarking against Samanvaya. | Keep and synchronize to public remote. |
| **Real Benchmark Manifest & Runner** | `samanvaya/validation/benchmark_real.py`, `lunar_core/cli.py` benchmark-real returning `DATA_REQUIRED` when rasters missing. | Basic manifest reader. | Local runner enforces authoritative data contract, streaming SHA-256 provenance, independent checkpoints, and claim gates. | Synchronize to public remote. |
| **Claim Gating** | `samanvaya/validation/claim_gate.py` with multi-category status (`PROVEN`, `SUPPORTED`, `PARTIAL`, `DATA_REQUIRED`, `FAILED`). | Basic checks. | Local claim gate prevents false claims and mandates independent checkpoints for `PROVEN`. | Synchronize to public remote. |
| **Real Product Inventory** | `samanvaya inventory --root data` classifying `AUTHORIZED_REAL`, `SYNTHETIC_FIXTURE`, `UNVERIFIED`, `INVALID`. | Basic discovery without strict 4-class categorization. | Local classification ensures only `AUTHORIZED_REAL` enters real benchmark. | Synchronize to public remote. |
| **IIRS Hyperspectral Pipeline** | Multi-band spectral continuum extraction; rejects arbitrary band collapsing with `IIRS_REPRESENTATION_UNCERTAIN`. | Basic spectral helper without registration gating. | Local pipeline ensures scientifically defensible 2-D representation. | Synchronize to public remote. |
| **Documentation & Protocols** | `docs/REAL_EXPERIMENT_PROTOCOL.md`, `docs/SIH_26166_SCORECARD.md`, `docs/REAL_DATASET_SETUP.md`. | Earlier drafts. | Complete scientific protocol and SIH 26166 traceability scorecard. | Synchronize to public remote. |

---

## 3. Required Synchronization Action

1. Verify all local scientific fixes pass unit and integration tests (163+ tests).
2. Commit all updated documentation, inventory, spatial distribution, and IIRS modules to `hardening-pass`.
3. Push `hardening-pass` to `origin/hardening-pass`.
4. Create or update PR against `main` so public tree contains the authoritative implementation.
