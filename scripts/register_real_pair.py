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
from lunar_core.data_io.mission_catalog import inspect_product, resolve_product_label
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


def _pds4_layout(image_path: Path, label_path: Path) -> tuple[int, int, int, np.dtype]:
    """Read the supported detached PDS4 2-D layout without allocating pixels."""
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

    expected_bytes = file_offset + lines * samples * dtype.itemsize
    if image_path.stat().st_size < expected_bytes:
        raise ValueError(
            f"PDS4 image is truncated: expected at least {expected_bytes} bytes, found {image_path.stat().st_size}"
        )
    return lines, samples, file_offset, dtype


def open_pds4_memmap(image_path: Path, label_path: Path) -> np.memmap:
    """Open a detached 2-D PDS4 image lazily for windowed processing."""
    lines, samples, file_offset, dtype = _pds4_layout(image_path, label_path)
    return np.memmap(image_path, dtype=dtype, mode="r", offset=file_offset, shape=(lines, samples), order="C")


def read_pds4_image(image_path: Path, label_path: Path) -> np.ndarray:
    """Read a common detached PDS4 2-D image using its label metadata."""
    data = np.asarray(open_pds4_memmap(image_path, label_path))
    return data.astype(np.float32)


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
    resolution = resolve_product_label(image_path)
    if not resolution.resolved or resolution.path is None:
        return 1.0, None
    sun, gsd, modality = PlanetaryRasterReader.parse_pds4_metadata(
        resolution.path, allowed_dir=image_path.parent
    )
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
            resolution = resolve_product_label(image_path)
            if not resolution.resolved or resolution.path is None:
                raise FileNotFoundError(
                    resolution.message or f"No PDS4 XML label found for {image_path.name}"
                )
            label = resolution.path
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


def _sun_angles_for_product(product) -> Optional[SunAngles]:
    if product.sun_azimuth_deg is None or product.sun_elevation_deg is None:
        return None
    return SunAngles(
        azimuth_deg=product.sun_azimuth_deg,
        elevation_deg=product.sun_elevation_deg,
    )


def register_pair(
    raw_dir: str | Path = "data/real/raw",
    output_dir: str | Path = "data/real/results",
    source: Optional[str] = None,
    target: Optional[str] = None,
    site: str = "unspecified",
    tile_threshold: int = 4096,
    tile_size: int = 1024,
    overlap: int = 128,
) -> int:
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    raw_dir = Path(raw_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = find_product(raw_dir, source, "source")
    target_path = find_product(raw_dir, target, "target")
    source_product = inspect_product(source_path, root_dir=source_path.parent)
    target_product = inspect_product(target_path, root_dir=target_path.parent)
    source_modality = _modality_for_product(source_product)
    target_modality = _modality_for_product(target_product)
    if source_modality == SensorModality.IIRS and (source_product.band_count or 0) > 1:
        raise RuntimeError("IIRS spectral cubes are catalog-aware but not yet supported by the 2-D registration runner.")
    if target_modality == SensorModality.IIRS and (target_product.band_count or 0) > 1:
        raise RuntimeError("IIRS spectral cubes are catalog-aware but not yet supported by the 2-D registration runner.")
    source_large_geotiff = source_path.suffix.lower() in {".tif", ".tiff"} and max(
        source_product.height or 0, source_product.width or 0
    ) >= args.tile_threshold
    target_large_geotiff = target_path.suffix.lower() in {".tif", ".tiff"} and max(
        target_product.height or 0, target_product.width or 0
    ) >= args.tile_threshold
    source_large_pds = source_path.suffix.lower() == ".img" and max(
        source_product.height or 0, source_product.width or 0
    ) >= args.tile_threshold
    target_large_pds = target_path.suffix.lower() == ".img" and max(
        target_product.height or 0, target_product.width or 0
    ) >= args.tile_threshold

    source = target = None
    source_masked = target_masked = 0
    source_gsd = source_product.gsd_m or 1.0
    target_gsd = target_product.gsd_m or 1.0
    source_sun = _sun_angles_for_product(source_product)
    target_sun = _sun_angles_for_product(target_product)
    source_shape = (source_product.height or 0, source_product.width or 0)
    target_shape = (target_product.height or 0, target_product.width or 0)

    both_window_readable = source_path.suffix.lower() in {".tif", ".tiff", ".img"} and target_path.suffix.lower() in {".tif", ".tiff", ".img"}
    if both_window_readable and (source_large_geotiff or target_large_geotiff or source_large_pds or target_large_pds):
        source_input = source_path
        target_input = target_path
        if source_path.suffix.lower() == ".img":
            source_label = resolve_product_label(source_path)
            if not source_label.resolved or source_label.path is None:
                raise FileNotFoundError(source_label.message or f"No PDS4 XML label for {source_path.name}")
            source_input = open_pds4_memmap(source_path, source_label.path)
        if target_path.suffix.lower() == ".img":
            target_label = resolve_product_label(target_path)
            if not target_label.resolved or target_label.path is None:
                raise FileNotFoundError(target_label.message or f"No PDS4 XML label for {target_path.name}")
            target_input = open_pds4_memmap(target_path, target_label.path)
        tiled = PlanetaryTileProcessor(tile_size=args.tile_size, overlap=args.overlap)
        tiled_result = tiled.process(source_input, target_input, estimate_coarse_overlap=False)
        total_matches = tiled_result.metrics.total_matches
        inliers = tiled_result.global_inliers
        matrix = tiled_result.global_homography
        matcher_path = "dense_loftr_or_classical_rift_per_tile_from_paths"
        processing_time_ms = tiled_result.processing_time_s * 1000.0
    else:
        source, source_masked = read_product(source_path, source_modality)
        target, target_masked = read_product(target_path, target_modality)
        source_gsd, target_gsd = source.gsd_meters, target.gsd_meters
        source_sun, target_sun = source.sun_angles, target.sun_angles
        source_shape, target_shape = source.shape, target.shape
        result = LunarCorePipeline().register(
            ref_image=target.data, target_image=source.data,
            ref_sun=target_sun, target_sun=source_sun,
            ref_gsd=target_gsd, target_gsd=source_gsd,
        )
        total_matches, inliers, matrix = len(result.matches), result.inliers, result.transform_matrix
        matcher_path, processing_time_ms = result.matcher_path, result.metrics.processing_time_ms

    report = EvaluationEngine.generate_report(
        total_matches=total_matches, inliers=inliers, homography=matrix,
        image_shape=target_shape, processing_time_ms=processing_time_ms,
    )
    report.export_json(output_dir / "evaluation_report.json")
    report.export_csv(output_dir / "evaluation_report.csv")
    provenance = {
        "dataset_class": "real_mission_data",
        "source": {
            "mission": source_product.mission,
            "instrument": source_product.instrument or source_modality.value,
            "product_id": source_product.product_id or source_path.stem,
            "file": str(source_path),
            "label": str(source_product.label_path) if source_product.label_path else None,
            "gsd_m": source_gsd,
            "sun_azimuth_deg": source_sun.azimuth_deg if source_sun else None,
            "sun_elevation_deg": source_sun.elevation_deg if source_sun else None,
            "masked_pixels": source_masked,
        },
        "target": {
            "mission": target_product.mission,
            "instrument": target_product.instrument or target_modality.value,
            "product_id": target_product.product_id or target_path.stem,
            "file": str(target_path),
            "label": str(target_product.label_path) if target_product.label_path else None,
            "gsd_m": target_gsd,
            "sun_azimuth_deg": target_sun.azimuth_deg if target_sun else None,
            "sun_elevation_deg": target_sun.elevation_deg if target_sun else None,
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
    print(f"Source: {source_path.name} ({source_shape}, {source_gsd:g} m/px)")
    print(f"Target: {target_path.name} ({target_shape}, {target_gsd:g} m/px)")
    print(f"Masked nodata pixels: source={source_masked}, target={target_masked}")
    print(f"Matcher path: {matcher_path}")
    print(f"RMSE: {report.rmse_pixels:.4f} px; inliers: {report.inlier_count}")
    print(f"Reports: {output_dir / 'evaluation_report.json'}, {output_dir / 'evaluation_report.csv'}")
    print(f"Provenance: {output_dir / 'provenance.json'}")
    return 0


def run(args: argparse.Namespace) -> int:
    return register_pair(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        source=args.source,
        target=args.target,
        site=args.site,
        tile_threshold=args.tile_threshold,
        tile_size=args.tile_size,
        overlap=args.overlap,
    )


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
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()