"""Register a downloaded mission-product raster pair."""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from importlib.metadata import version as package_version, PackageNotFoundError
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from lunar_core.data_io import PlanetaryRasterReader, PlanetaryTileProcessor
from lunar_core.data_io.mission_catalog import inspect_product
from lunar_core.evaluation.metrics import EvaluationEngine
from lunar_core.models import GeoRaster, SensorModality, SunAngles
from lunar_core.pipeline import LunarCorePipeline


def _software_version() -> str:
    try:
        return package_version("samanvaya")
    except PackageNotFoundError:
        return "uninstalled-source-tree"

IMAGE_SUFFIXES = {".tif", ".tiff", ".img"}


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def read_pds4_image(image_path: Path, label_path: Path) -> np.ndarray:
    """Read a common detached PDS4 2-D image using its label metadata."""
    try:
        from defusedxml import ElementTree
        root = ElementTree.parse(label_path).getroot()
    except Exception as exc:
        raise RuntimeError(f"Unable to safely parse PDS4 label: {label_path}") from exc

    nodes = list(root.iter())

    def values(name: str) -> list[str]:
        return [node.text.strip() for node in nodes if _local_name(node.tag) == name and node.text]

    file_offset = 0
    offsets = values("offset")
    if offsets:
        file_offset = int(float(re.sub(r"[^0-9.+-]", "", offsets[0])))
    dimensions = [int(value) for value in values("elements")]
    if len(dimensions) < 2:
        raise ValueError(f"PDS4 label has no 2-D image dimensions: {label_path}")
    lines, samples = dimensions[-2], dimensions[-1]
    data_type = (values("data_type") or ["MSB_INTEGER"])[0].upper()
    bits_value = values("bits")
    if bits_value:
        bits = int(bits_value[0])
    elif "BYTE" in data_type:
        bits = 8
    elif "WORD" in data_type:
        bits = 16
    elif "REAL" in data_type:
        bits = 32
    else:
        bits = 16
    if "REAL" in data_type:
        dtype = np.dtype(">f4" if bits == 32 and "MSB" in data_type else "<f4")
    elif bits in (8, 16, 32, 64):
        signed = "UNSIGNED" not in data_type
        kind = "i" if signed else "u"
        endian = ">" if "MSB" in data_type else "<"
        dtype = np.dtype(f"{endian}{kind}{bits // 8}")
    else:
        raise ValueError(f"Unsupported PDS4 sample width/type: {bits} bits, {data_type}")

    count = lines * samples
    with image_path.open("rb") as stream:
        stream.seek(file_offset)
        data = np.fromfile(stream, dtype=dtype, count=count)
    if data.size != count:
        raise ValueError(f"PDS4 image is truncated: expected {count} samples, found {data.size}")
    return data.reshape((lines, samples)).astype(np.float32)


def find_product(raw_dir: Path, requested: Optional[str], role: str) -> Path:
    if requested:
        path = Path(requested).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{role} product not found: {path}")
        return path
    candidates = sorted(
        path for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(candidates) != 1:
        names = ", ".join(path.name for path in candidates) or "none"
        raise RuntimeError(f"Expected exactly one {role} image in {raw_dir}; found: {names}. Use an explicit path.")
    return candidates[0]


def product_metadata(image_path: Path) -> Tuple[float, Optional[SunAngles]]:
    labels = sorted(image_path.parent.glob("*.xml"))
    label = next((path for path in labels if image_path.stem.lower() in path.stem.lower()), labels[0] if labels else None)
    if label is None:
        return 1.0, None
    sun, gsd, modality = PlanetaryRasterReader.parse_pds4_metadata(label, allowed_dir=image_path.parent)
    return gsd, sun if modality != SensorModality.SYNTHETIC else None


def black_edge_mask(data: np.ndarray, nodata_val: Optional[float] = None) -> np.ndarray:
    finite = np.isfinite(data)
    if not finite.any():
        return ~finite
    scale = float(np.nanpercentile(data[finite], 99.0))
    black = finite & (np.abs(data) <= max(abs(scale) * 1e-6, 1e-6))
    if nodata_val is not None:
        black |= finite & np.isclose(data, float(nodata_val))
    mask = ~finite
    for axis in (0, 1):
        for index in range(data.shape[axis]):
            edge = np.take(black, index, axis=axis)
            if float(edge.mean()) < 0.98:
                break
            if axis == 0:
                mask[index, :] = True
            else:
                mask[:, index] = True
        for index in range(1, data.shape[axis] + 1):
            edge = np.take(black, -index, axis=axis)
            if float(edge.mean()) < 0.98:
                break
            if axis == 0:
                mask[-index, :] = True
            else:
                mask[:, -index] = True
    return mask


def read_product(image_path: Path, modality: SensorModality) -> Tuple[GeoRaster, int]:
    gsd, sun = product_metadata(image_path)
    try:
        if image_path.suffix.lower() == ".img":
            labels = sorted(image_path.parent.glob("*.xml"))
            label = next((path for path in labels if image_path.stem.lower() in path.stem.lower()), None)
            if label is None:
                raise FileNotFoundError(f"No PDS4 XML label found for {image_path.name}")
            data = read_pds4_image(image_path, label)
            raster = GeoRaster(data=data, modality=modality, gsd_meters=gsd, sun_angles=sun)
        else:
            raster = PlanetaryRasterReader.read_geotiff(
                image_path, modality=modality, gsd_fallback=gsd,
                allowed_dir=image_path.parent, sun_angles=sun,
            )
    except Exception as exc:
        if image_path.suffix.lower() == ".img":
            raise RuntimeError(
                f"Could not decode detached PDS4 image {image_path} using its XML label. "
                "Verify that the label describes a supported 2-D image array."
            ) from exc
        raise

    invalid = black_edge_mask(np.asarray(raster.data, dtype=np.float32), raster.nodata_val)
    masked_count = int(invalid.sum())
    valid_rows = ~invalid.all(axis=1)
    valid_cols = ~invalid.all(axis=0)
    data = raster.data[np.ix_(valid_rows, valid_cols)]
    if min(data.shape) < 32:
        raise ValueError(f"Valid footprint after nodata masking is too small: {data.shape}")
    return GeoRaster(
        data=data, modality=raster.modality, gsd_meters=raster.gsd_meters,
        sun_angles=raster.sun_angles, transform=raster.transform,
        crs=raster.crs, nodata_val=raster.nodata_val,
    ), masked_count


def _modality_for_product(product) -> SensorModality:
    instrument = (product.instrument or "").upper()
    mapping = {
        "OHRC": SensorModality.OHRC,
        "TMC-2": SensorModality.TMC2,
        "IIRS": SensorModality.IIRS,
        "NAC": SensorModality.LRO_NAC,
        "LROC": SensorModality.LRO_NAC,
        "TC": SensorModality.SYNTHETIC,
    }
    return mapping.get(instrument, SensorModality.SYNTHETIC)


def run(args: argparse.Namespace) -> None:
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    raw_dir = Path(args.raw_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = find_product(raw_dir, args.source, "source")
    target_path = find_product(raw_dir, args.target, "target")
    source_product = inspect_product(source_path, root_dir=source_path.parent)
    target_product = inspect_product(target_path, root_dir=target_path.parent)
    source_modality = _modality_for_product(source_product)
    target_modality = _modality_for_product(target_product)
    source, source_masked = read_product(source_path, source_modality)
    target, target_masked = read_product(target_path, target_modality)

    if max(source.shape + target.shape) >= args.tile_threshold:
        tiled = PlanetaryTileProcessor(tile_size=args.tile_size, overlap=args.overlap)
        tiled_result = tiled.process(source.data, target.data, estimate_coarse_overlap=False)
        total_matches = tiled_result.metrics.total_matches
        inliers = tiled_result.global_inliers
        matrix = tiled_result.global_homography
        matcher_path = "dense_loftr_or_classical_rift_per_tile"
        processing_time_ms = tiled_result.processing_time_s * 1000.0
    else:
        result = LunarCorePipeline().register(
            ref_image=target.data, target_image=source.data,
            ref_sun=target.sun_angles, target_sun=source.sun_angles,
            ref_gsd=target.gsd_meters, target_gsd=source.gsd_meters,
        )
        total_matches, inliers, matrix = len(result.matches), result.inliers, result.transform_matrix
        matcher_path, processing_time_ms = result.matcher_path, result.metrics.processing_time_ms

    report = EvaluationEngine.generate_report(
        total_matches=total_matches, inliers=inliers, homography=matrix,
        image_shape=target.shape, processing_time_ms=processing_time_ms,
    )
    report.export_json(output_dir / "evaluation_report.json")
    report.export_csv(output_dir / "evaluation_report.csv")
    provenance = {
        "dataset_class": "real_mission_data",
        "source": {
            "mission": source_product.mission,
            "instrument": source_product.instrument or source.modality.value,
            "product_id": source_product.product_id or source_path.stem,
            "file": str(source_path),
            "label": str(source_product.label_path) if source_product.label_path else None,
            "gsd_m": source.gsd_meters,
            "sun_azimuth_deg": source.sun_angles.azimuth_deg if source.sun_angles else None,
            "sun_elevation_deg": source.sun_angles.elevation_deg if source.sun_angles else None,
            "masked_pixels": source_masked,
        },
        "target": {
            "mission": target_product.mission,
            "instrument": target_product.instrument or target.modality.value,
            "product_id": target_product.product_id or target_path.stem,
            "file": str(target_path),
            "label": str(target_product.label_path) if target_product.label_path else None,
            "gsd_m": target.gsd_meters,
            "sun_azimuth_deg": target.sun_angles.azimuth_deg if target.sun_angles else None,
            "sun_elevation_deg": target.sun_angles.elevation_deg if target.sun_angles else None,
            "masked_pixels": target_masked,
        },
        "pipeline": {
            "matcher": matcher_path,
            "fallback_used": "rift" in matcher_path.lower(),
            "photometric_correction": "enabled",
            "tile_threshold": args.tile_threshold,
            "tile_size": args.tile_size,
            "overlap": args.overlap,
            "software_version": _software_version(),
        },
        "execution": {
            "timestamp_utc": started_at,
            "runtime_seconds": time.perf_counter() - started,
            "real_rmse_pixels": report.rmse_pixels,
            "inlier_count": report.inlier_count,
            "ground_truth_available": False,
            "metric_note": "Reprojection/consensus metric; no independent ground truth supplied.",
        },
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Site: {args.site}")
    print(f"Source: {source_path.name} ({source.shape}, {source.gsd_meters:g} m/px)")
    print(f"Target: {target_path.name} ({target.shape}, {target.gsd_meters:g} m/px)")
    print(f"Masked nodata pixels: source={source_masked}, target={target_masked}")
    print(f"Matcher path: {matcher_path}")
    print(f"RMSE: {report.rmse_pixels:.4f} px; inliers: {report.inlier_count}")
    print(f"Reports: {output_dir / 'evaluation_report.json'}, {output_dir / 'evaluation_report.csv'}")
    print(f"Provenance: {output_dir / 'provenance.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a mission-product source/target pair.")
    parser.add_argument("--raw-dir", default="data/real/raw")
    parser.add_argument("--output-dir", default="data/real/results")
    parser.add_argument("--source", "--chandrayaan", dest="source")
    parser.add_argument("--target", "--lro", dest="target")
    parser.add_argument("--site", default="unspecified")
    parser.add_argument("--tile-threshold", type=int, default=4096)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    run(parser.parse_args())


if __name__ == "__main__":
    main()