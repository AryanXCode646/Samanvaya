# Security Policy

## Supported Versions

Samanvaya is actively maintained for ISRO Smart India Hackathon PS 26166 and downstream lunar remote sensing applications.

| Version | Supported          | Python Target |
| ------- | ------------------ | ------------- |
| 1.x     | :white_check_mark: | 3.10, 3.11    |

## Defensive Architecture & Built-in Protections

Samanvaya handles untrusted planetary remote sensing datasets and incorporates the following built-in security defenses:

1. **XML External Entity (XXE) Prevention**:
   - All PDS4 label XML files are parsed using `defusedxml` with external entity resolution explicitly disabled (`resolve_entities=False`).
2. **Path Traversal Shielding**:
   - File access paths undergo strict canonicalization and boundary checks (`os.path.realpath`, `resolve()`).
   - Null bytes (`\0`) and directory traversal sequences (`..`) in user-supplied filenames are rejected before filesystem access.
3. **Decompression Bomb Protection**:
   - Ingested raster dimensions are bounded (maximum $30,000 \times 30,000$ pixels) with a $4\text{ GB}$ maximum uncompressed buffer limit, rejecting maliciously crafted compressed GeoTIFFs before memory allocation.
4. **Out-of-Core Memory Ceilings**:
   - Windowed tile streaming via Rasterio prevents denial-of-service memory exhaustion during gigapixel swath alignment.

## Reporting a Vulnerability

If you discover a security vulnerability within Samanvaya, please report it responsibly:

1. Open a confidential advisory or email the maintainer directly at `mr.ashishsinghbora@gmail.com`.
2. Please include detailed steps to reproduce the issue, including sample input files if applicable.
3. You can expect an initial acknowledgment within 48 hours and regular updates until a fix is published.

