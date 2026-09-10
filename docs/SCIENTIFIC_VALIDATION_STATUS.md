# Scientific Validation Status

This repository remains in a conservative evidence-boundary state. The code contains a real pipeline, metadata-aware mission adapters, and a validation framework, but the dataset coverage needed for a full real lunar-mission claim is not yet present in this checkout.

## Claim audit

| Claim | Current evidence | Dataset | Independent? | Status |
|---|---|---|---|---|
| Multi-mission mission adapter support | Mission catalog and metadata parsing exist | Synthetic + metadata fixtures | No | PARTIAL |
| OHRC ingestion path | Adapter and PDS4 metadata parsing exist | Real metadata fixtures, not full image validation | No | PARTIAL |
| TMC-2 ingestion path | Adapter and metadata handling exist | Real metadata fixtures, not full image validation | No | PARTIAL |
| IIRS ingestion | Catalog-aware flow exists; spectral-cube handling remains limited | Metadata only | No | PARTIAL |
| LROC NAC ingestion | Adapter scaffolding exists | Metadata only | No | PARTIAL |
| SELENE/KAGUYA ingestion | Adapter scaffolding exists | Metadata only | No | PARTIAL |
| Real registration accuracy claim | Not present without independent ground truth | Pending data | N/A | NOT CLAIMED |
| Synthetic benchmark accuracy | Code and tests support it | Synthetic | N/A | PROVEN WITHIN SCOPE |
| Scientific mission validation | Not demonstrated with real data | Pending mission datasets | N/A | BLOCKED BY DATA AVAILABILITY |

## Bottom line

The repository is not claiming real-mission validation without independent groundwater-truth data. The current implementation is defensible as a structured, reproducible, and evidence-aware validation framework, but not as a full end-to-end real-mission scientific result.
