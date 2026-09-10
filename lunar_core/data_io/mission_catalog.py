"""Mission-product discovery and metadata-only catalog generation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from lunar_core.data_io.mission_product import IdentificationMethod, MissionProduct, ProductStatus
from lunar_core.data_io.product_identity import (
    LabelAssociationStatus,
    identify_mission_instrument,
    resolve_product_label,
)
from lunar_core.data_io.raster_reader import PlanetaryRasterReader, sanitize_path
from lunar_core.models import SensorModality

IMAGE_SUFFIXES = {".tif", ".tiff", ".img", ".qub"}
MANIFEST_FIELDS = list(MissionProduct.__dataclass_fields__.keys())


def _tag_name(node: Any) -> str:
    tag = getattr(node, "tag", "")
    if not isinstance(tag, str):
        return str(tag).lower()
    return tag.rsplit("}", 1)[-1].lower()


def _local_values(root: Any, name: str) -> list[str]:
    target = name.lower()
    return [
        node.text.strip()
        for node in root.iter()
        if node.text and _tag_name(node) == target
    ]


def _first_value(root: Any, *names: str) -> Optional[str]:
    for name in names:
        values = _local_values(root, name)
        if values:
            return values[0]
    return None


def _read_label(label_path: Path) -> Any:
    from defusedxml import ElementTree

    return ElementTree.parse(label_path).getroot()


def _header_metadata(image_path: Path, label_root: Any) -> dict[str, Any]:
    if image_path.suffix.lower() in {".img", ".qub"}:
        dimensions = [int(value) for value in _local_values(label_root, "elements")]
        dtype_values = _local_values(label_root, "data_type")
        return {
            "width": dimensions[-1] if len(dimensions) >= 1 else None,
            "height": dimensions[-2] if len(dimensions) >= 2 else None,
            "band_count": dimensions[-3] if len(dimensions) >= 3 else None,
            "dtype": dtype_values[0] if dtype_values else None,
        }

    import rasterio

    with rasterio.open(image_path) as dataset:
        return {
            "width": dataset.width,
            "height": dataset.height,
            "band_count": dataset.count,
            "dtype": dataset.dtypes[0] if dataset.dtypes else None,
            "crs": str(dataset.crs) if dataset.crs else None,
        }


def _center_coordinates(label_root: Any) -> tuple[Optional[float], Optional[float]]:
    """Average explicit latitude/longitude corner fields when a label provides them."""
    latitudes: list[float] = []
    longitudes: list[float] = []
    for node in label_root.iter():
        if not node.text:
            continue
        name = node.tag.rsplit("}", 1)[-1].lower()
        try:
            value = float(node.text.strip())
        except ValueError:
            continue
        if "latitude" in name:
            latitudes.append(value)
        elif "longitude" in name:
            longitudes.append(value)
    return (
        sum(latitudes) / len(latitudes) if latitudes else None,
        sum(longitudes) / len(longitudes) if longitudes else None,
    )


def _footprint_from_coordinates(label_root: Any) -> Optional[list[tuple[float, float]]]:
    """Build a conservative lat/lon bounding polygon from structured corner fields."""
    latitudes: list[float] = []
    longitudes: list[float] = []
    for node in label_root.iter():
        if not node.text:
            continue
        name = _tag_name(node)
        try:
            value = float(node.text.strip())
        except ValueError:
            continue
        if "latitude" in name:
            latitudes.append(value)
        elif "longitude" in name:
            longitudes.append(value)
    if len(latitudes) < 2 or len(longitudes) < 2:
        return None
    min_lat, max_lat = min(latitudes), max(latitudes)
    min_lon, max_lon = min(longitudes), max(longitudes)
    return [
        (min_lon, min_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
        (min_lon, max_lat),
    ]


def inspect_product(image_path: Path, root_dir: Optional[Path] = None) -> MissionProduct:
    """Inspect one product without allocating its raster pixels."""
    image_path = sanitize_path(image_path, allowed_dir=root_dir)
    resolution = resolve_product_label(image_path)
    label_path = resolution.path if resolution.resolved else None
    product = MissionProduct(
        mission=None,
        instrument=None,
        product_id=image_path.stem,
        image_path=image_path,
        label_path=label_path,
        file_format=image_path.suffix.lower().lstrip("."),
        identification_method=IdentificationMethod.UNKNOWN.value,
        label_association_method=resolution.method.value if resolution.method else None,
        status=ProductStatus.DISCOVERED,
    )
    detached = image_path.suffix.lower() in {".img", ".qub"}
    if label_path is None and detached:
        product.status = ProductStatus.INVALID
        product.validation_status = ProductStatus.INVALID.value
        product.validation_message = resolution.message or (
            "Detached product has no authoritative PDS4 XML label."
        )
        if resolution.status is LabelAssociationStatus.AMBIGUOUS:
            product.status = ProductStatus.INVALID
        return product

    try:
        label_root = _read_label(label_path) if label_path else None
        header = _header_metadata(image_path, label_root) if label_root is not None else _header_metadata(image_path, None)
        for key, value in header.items():
            setattr(product, key, value)
        if label_root is not None:
            product.product_id = _first_value(
                label_root, "product_id", "product_identifier", "logical_identifier", "product_name"
            ) or product.product_id
            mission, instrument, method = identify_mission_instrument(image_path, label_root)
            product.mission = mission
            product.instrument = instrument
            product.identification_method = method.value
            sun, gsd, modality = PlanetaryRasterReader.parse_pds4_metadata(
                label_path, allowed_dir=label_path.parent
            )
            product.gsd_m = gsd
            if _first_value(label_root, "pixel_resolution", "ground_sample_distance", "gsd", "resolution"):
                product.gsd_source = "pds4"
            elif modality != SensorModality.SYNTHETIC:
                product.gsd_source = "instrument_default"
            else:
                product.gsd_source = "unknown"
            if sun is not None:
                product.sun_azimuth_deg = sun.azimuth_deg
                product.sun_elevation_deg = sun.elevation_deg
                product.sun_geometry_source = "pds4"
            else:
                product.sun_geometry_source = "unknown"
            product.center_lat_deg, product.center_lon_deg = _center_coordinates(label_root)
            product.footprint = _footprint_from_coordinates(label_root)
            product.metadata_source = str(label_path)
            product.acquisition_time = next(iter(_local_values(label_root, "start_date_time")), None)
            product.processing_level = next(iter(_local_values(label_root, "processing_level")), None)
            product.product_type = next(iter(_local_values(label_root, "product_class")), None) or _first_value(
                label_root, "product_type"
            )
            if (product.band_count or 0) > 1:
                product.status = ProductStatus.PARTIAL
                product.validation_status = ProductStatus.PARTIAL.value
                product.validation_message = (
                    "Spectral cube metadata parsed; 2-D registration representation is not implemented."
                )
            else:
                product.status = ProductStatus.VALIDATED
                product.validation_status = ProductStatus.VALIDATED.value
    except Exception as exc:
        product.status = ProductStatus.INVALID
        product.validation_status = ProductStatus.INVALID.value
        product.validation_message = f"{type(exc).__name__}: {exc}"
    return product


def scan(root_dir: Path) -> list[MissionProduct]:
    """Recursively inspect candidate mission products, retaining failures."""
    root_dir = sanitize_path(root_dir)
    products: list[MissionProduct] = []
    for path in sorted(root_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            products.append(inspect_product(path, root_dir=root_dir))
    return products


def write_csv(products: Iterable[MissionProduct], output_path: Path) -> None:
    records = [product.to_dict() for product in products]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan mission products without loading raster pixels.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    products = scan(args.root)
    records = [product.to_dict() for product in products]
    if args.output:
        write_csv(products, args.output)
    else:
        print(json.dumps(records, indent=2))
    invalid = sum(product.status != "validated" for product in products)
    raise SystemExit(1 if invalid else 0)


if __name__ == "__main__":
    main()
