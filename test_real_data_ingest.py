#!/usr/bin/env python3
"""Verify ingestion of real planetary rasters and their PDS4 labels."""

from __future__ import annotations

import argparse
from pathlib import Path

from lunar_core.data_io.raster_reader import PlanetaryRasterReader

SUPPORTED_IMAGE_SUFFIXES = {".tif", ".tiff", ".img"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect real PDS4 raster products.")
    parser.add_argument("--data-dir", default="lunar_core/assets/real_data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        print(f"ERROR: real-data directory does not exist: {data_dir}")
        return 1

    images = sorted(
        path for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if not images:
        print(f"No supported raster products found in {data_dir}")
        print("Expected a GeoTIFF or detached .img plus a same-stem .xml label.")
        return 2

    failures = 0
    for image_path in images:
        label_path = image_path.with_suffix(".xml")
        print(f"\nProduct: {image_path}")
        if not label_path.is_file():
            print(f"ERROR: same-stem PDS4 label not found: {label_path}")
            failures += 1
            continue
        try:
            sun, gsd, modality = PlanetaryRasterReader.parse_pds4_metadata(
                label_path, allowed_dir=image_path.parent
            )
            raster = PlanetaryRasterReader.read_georaster(
                image_path,
                modality=modality,
                gsd_fallback=gsd,
                allowed_dir=image_path.parent,
                sun_angles=sun,
            )
            print(f"Shape: {raster.shape}")
            print(f"GSD: {raster.gsd_meters:g} m/px")
            print(f"Modality: {raster.modality.value}")
            print(f"Sun azimuth: {raster.sun_angles.azimuth_deg:g} degrees")
            print(f"Sun elevation: {raster.sun_angles.elevation_deg:g} degrees")
            print(f"CRS: {raster.crs}")
        except Exception as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}")
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
