"""Mission-product discovery and metadata-only catalog generation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from lunar_core.data_io.mission_product import MissionProduct
from lunar_core.data_io.raster_reader import PlanetaryRasterReader, sanitize_path
from lunar_core.models import SensorModality

IMAGE_SUFFIXES = {".tif", ".tiff", ".img", ".qub"}
MANIFEST_FIELDS = list(MissionProduct.__dataclass_fields__.keys())


def _local_values(root: Any, name: str) -> list[str]:
    return [
        node.text.strip()
        for node in root.iter()
        if node.text and node.tag.rsplit("}", 1)[-1].lower() == name.lower()
    ]


def _label_for(image_path: Path) -> Optional[Path]:
    exact = image_path.with_suffix(".xml")
    if exact.is_file():
        return exact
    candidates = sorted(image_path.parent.glob("*.xml"))
    return candidates[0] if len(candidates) == 1 else None


def _infer_mission_instrument(image_path: Path, label_root: Any) -> tuple[Optional[str], Optional[str]]:
    text = " ".join(
        node.text.strip()
        for node in label_root.iter()
        if node.text and node.text.strip()
    ).upper()
    name = image_path.name.upper()
    combined = f"{name} {text}"
    if "CHANDRAYAAN-2" in combined or "CHANDRAYAAN 2" in combined or "CH2_" in combined:
        mission = "Chandrayaan-2"
    elif "LUNAR RECONNAISSANCE ORBITER" in combined or "LRO" in combined:
        mission = "LRO"
    elif "SELENE" in combined or "KAGUYA" in combined:
        mission = "SELENE"
    else:
        mission = None

    if "OHRC" in combined or "HIGH RESOLUTION CAMERA" in combined:
        instrument = "OHRC"
    elif "TMC" in combined or "TERRAIN MAPPING CAMERA" in combined:
        instrument = "TMC-2"
    elif "IIRS" in combined or "IMAGING INFRARED SPECTROMETER" in combined:
        instrument = "IIRS"
    elif "NAC" in combined or "NARROW ANGLE CAMERA" in combined:
        instrument = "NAC"
    elif "LROC" in combined:
        instrument = "LROC"
    elif "TC" in combined:
        instrument = "TC"
    else:
        instrument = None
    return mission, instrument


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


def inspect_product(image_path: Path, root_dir: Optional[Path] = None) -> MissionProduct:
    """Inspect one product without allocating its raster pixels."""
    image_path = sanitize_path(image_path, allowed_dir=root_dir)
    label_path = _label_for(image_path)
    product = MissionProduct(
        mission=None,
        instrument=None,
        product_id=image_path.stem,
        image_path=image_path,
        label_path=label_path,
        file_format=image_path.suffix.lower().lstrip("."),
        status="validated",
    )
    if label_path is None and image_path.suffix.lower() in {".img", ".qub"}:
        product.status = "invalid"
        product.validation_message = "Detached product has no same-stem XML label."
        return product

    try:
        label_root = _read_label(label_path) if label_path else None
        header = _header_metadata(image_path, label_root) if label_root is not None else _header_metadata(image_path, None)
        for key, value in header.items():
            setattr(product, key, value)
        if label_root is not None:
            product.mission, product.instrument = _infer_mission_instrument(image_path, label_root)
            sun, gsd, modality = PlanetaryRasterReader.parse_pds4_metadata(
                label_path, allowed_dir=label_path.parent
            )
            product.gsd_m = gsd
            product.sun_azimuth_deg = sun.azimuth_deg
            product.sun_elevation_deg = sun.elevation_deg
            product.center_lat_deg, product.center_lon_deg = _center_coordinates(label_root)
            product.metadata_source = str(label_path)
            if product.instrument is None and modality != SensorModality.SYNTHETIC:
                product.instrument = modality.value
            product.acquisition_time = next(iter(_local_values(label_root, "start_date_time")), None)
            product.processing_level = next(iter(_local_values(label_root, "processing_level")), None)
            product.product_type = next(iter(_local_values(label_root, "product_class")), None)
    except Exception as exc:
        product.status = "invalid"
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
