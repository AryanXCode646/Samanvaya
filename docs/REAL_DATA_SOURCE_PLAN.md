# Real Data Source Plan

This repository is now configured to accept the exact real-data sources you identified, while keeping all scientific claims conservative until the actual mission products and independent control points are present.

## Official mission sources

- Chandrayaan-2 orbiter optical payloads (OHRC, TMC-2, IIRS):
  - https://chmapbrowse.issdc.gov.in/
- LRO NAC reference imagery:
  - https://lroc.im-ldi.com/images/downloads/
  - https://quickmap.lroc.im-ldi.com/
- SELENE reference imagery:
  - https://darts.isas.jaxa.jp/app/pdap/selene/

## Data handling policy

- The repo will ingest only products that are actually present locally or explicitly supplied by the user.
- Product IDs and URLs remain configurable via the dataset manifest rather than hard-coded assumptions.
- No real validation result is claimed until independent ground truth exists for the selected pair.
- Synthetic and real-data results remain separate in the report and validation matrix.

## Planned real benchmark pairs

- OHRC ↔ TMC-2
- OHRC ↔ IIRS
- TMC-2 ↔ IIRS
- OHRC ↔ LROC NAC
- TMC-2 ↔ LROC NAC
- SELENE ↔ LROC NAC

## Required next inputs

To complete genuine validation, the following must be supplied:

1. exact product IDs or download URLs for each mission image,
2. local file paths or archived bundles in the workspace,
3. independent control-point JSON files for ground-truth evaluation,
4. exact mission pair selection and overlap metadata.

Until those are present, the system remains in a conservative `PENDING_DATA` state, which is the scientifically defensible state.
