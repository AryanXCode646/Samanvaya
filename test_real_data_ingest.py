#!/usr/bin/env python3
"""Verify ingestion of real planetary rasters and their PDS4 labels."""

from __future__ import annotations

import argparse
from pathlib import Path

from lunar_core.data_io.mission_catalog import inspect_product, resolve_product_label
from lunar_core.data_io.raster_reader import PlanetaryRasterReader

SUPPORTED_IMAGE_SUFFIXES = {".tif", ".tiff", ".img", ".qub"}


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
        print(f"\nProduct: {image_path}")
        resolution = resolve_product_label(image_path)
        if not resolution.resolved or resolution.path is None:
            print(f"ERROR: {resolution.message or 'canonical PDS4 label resolution failed'}")
            failures += 1
            continue
        label_path = resolution.path
        try:
            product = inspect_product(image_path, root_dir=data_dir)
            status = product.status.value if hasattr(product.status, "value") else product.status
            print(f"Status: {status}")
            print(f"Mission: {product.mission or 'unknown'}")
            print(f"Instrument: {product.instrument or 'unknown'}")
            print(f"Shape: ({product.height}, {product.width})")
            print(f"Bands: {product.band_count or 1}")
            print(f"GSD: {product.gsd_m if product.gsd_m is not None else 'unknown'} m/px")
            print(f"Sun azimuth: {product.sun_azimuth_deg if product.sun_azimuth_deg is not None else 'unknown'} degrees")
            print(f"Sun elevation: {product.sun_elevation_deg if product.sun_elevation_deg is not None else 'unknown'} degrees")
            if image_path.suffix.lower() == ".img":
                mapped = PlanetaryRasterReader.open_pds4_memmap(image_path, label_path, allowed_dir=data_dir)
                print(f"Windowed 2-D access: {mapped.shape}")
            elif image_path.suffix.lower() == ".qub":
                mapped, shape = PlanetaryRasterReader.open_pds4_spectral_memmap(image_path, label_path, allowed_dir=data_dir)
                print(f"Windowed spectral access: {shape}; first sample={float(mapped[0, 0, 0])}")
            if status in {"invalid", "unsupported"}:
                failures += 1
        except Exception as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}")
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
