"""
Planetary Data Ingestion Driver (GDAL/Rasterio GeoTIFF and PDS4 Reader).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union
import os
import urllib.parse
import urllib.request
import numpy as np
import rasterio

try:
    from lxml import etree as lxml_ET
    LXML_AVAILABLE = True
except ImportError:
    lxml_ET = None
    LXML_AVAILABLE = False

try:
    import defusedxml.ElementTree as defused_ET
    from defusedxml.common import DefusedXmlException
    DEFUSED_AVAILABLE = True
except ImportError:
    import xml.etree.ElementTree as defused_ET
    class DefusedXmlException(Exception):
        pass
    DEFUSED_AVAILABLE = False

from lunar_core.models import GeoRaster, SensorModality, SunAngles

# Cybersecurity & Memory Safety Thresholds
# OHRC Level-2 products can be pushbroom strips wider than 30,000 pixels.
# The 4 GiB uncompressed ceiling remains the primary allocation guard.
MAX_RASTER_DIMENSION = 100000         # Maximum supported single raster dimension
MAX_UNCOMPRESSED_BYTES = 4 * 1024**3  # 4 GiB hard memory safety limit


def sanitize_path(input_path: Union[str, Path], allowed_dir: Optional[Path] = None) -> Path:
    """
    Sanitizes file paths and URIs against directory traversal attacks ('../' escapes),
    null bytes, and escapes outside designated directories.
    Handles standard paths and file:// URIs.
    """
    path_str = str(input_path).strip()
    if "\x00" in path_str:
        raise ValueError("Security violation: null byte detected in file path")

    # Parse and unwrap file:// URI schemes
    if path_str.startswith("file:"):
        parsed = urllib.parse.urlparse(path_str)
        if os.name == "nt":
            combined = f"{parsed.netloc}{parsed.path}"
            path_str = urllib.request.url2pathname(combined).lstrip("\\/")
            path_str = os.path.normpath(path_str)
        else:
            path_str = urllib.parse.unquote(parsed.path)
            if parsed.netloc and parsed.netloc != "localhost":
                path_str = f"/{parsed.netloc}{path_str}"

    # Handle percent-encoded characters e.g. %2e%2e%2f -> ../
    unquoted = urllib.parse.unquote(path_str)
    if "\x00" in unquoted:
        raise ValueError("Security violation: null byte detected in file path")

    resolved = Path(unquoted).resolve()

    if allowed_dir is not None:
        allowed = allowed_dir.resolve()
        try:
            resolved.relative_to(allowed)
        except ValueError:
            raise PermissionError(f"Security violation: path traversal outside allowed directory '{allowed}'")
    return resolved


class PlanetaryRasterReader:
    """
    Reads planetary imagery from standard GeoTIFF formats and PDS4 product labels.
    Hardened against XXE injection, decompression bombs, and directory traversal.
    """

    @staticmethod
    def read_geotiff(
        filepath: Union[str, Path],
        modality: SensorModality = SensorModality.SYNTHETIC,
        gsd_fallback: float = 1.0,
        allowed_dir: Optional[Path] = None,
        sun_angles: Optional[SunAngles] = None,
    ) -> GeoRaster:
        """
        Ingests georeferenced GeoTIFF raster and extracts spatial resolution and CRS.
        Enforces decompression bomb and memory allocation checks before raster reading.
        """
        path = sanitize_path(filepath, allowed_dir=allowed_dir)
        if not path.exists():
            raise FileNotFoundError(f"GeoTIFF file not found: {path}")

        if path.suffix.lower() == ".img":
            label_path = path.with_suffix(".xml")
            if not label_path.exists():
                raise FileNotFoundError(f"PDS4 XML label not found for detached image: {label_path}")
            return PlanetaryRasterReader._read_pds4_image(
                path,
                label_path,
                modality=modality,
                gsd_fallback=gsd_fallback,
                allowed_dir=allowed_dir,
                sun_angles=sun_angles,
            )

        with rasterio.open(str(path)) as src:
            # Shield against Decompression Bomb Denial of Service (DoS)
            if src.width > MAX_RASTER_DIMENSION or src.height > MAX_RASTER_DIMENSION:
                raise ValueError(
                    f"Decompression bomb rejected: Raster dimensions ({src.width}x{src.height}) "
                    f"exceed security ceiling ({MAX_RASTER_DIMENSION}x{MAX_RASTER_DIMENSION}). "
                    "Use PlanetaryTileProcessor for out-of-core windowed processing."
                )

            itemsize = np.dtype(src.dtypes[0]).itemsize if src.dtypes else 4
            estimated_bytes = src.width * src.height * src.count * itemsize
            if estimated_bytes > MAX_UNCOMPRESSED_BYTES:
                raise MemoryError(
                    f"Decompression bomb rejected: Buffer ({estimated_bytes / (1024**2):.1f} MB) "
                    f"exceeds safe threshold ({MAX_UNCOMPRESSED_BYTES / (1024**2):.1f} MB). "
                    "Use PlanetaryTileProcessor for out-of-core windowed processing."
                )

            data = src.read(1).astype(np.float32)
            transform = src.transform
            crs = str(src.crs) if src.crs else "IAU2000:30100"
            nodata = src.nodata

            # Pixel dimensions in meters from affine transform
            res_x = abs(transform[0])
            res_y = abs(transform[4])
            gsd = float((res_x + res_y) / 2.0) if (res_x > 0 and res_y > 0) else gsd_fallback

            # Read optional solar metadata tags
            if sun_angles is not None:
                sun = sun_angles
            else:
                tags = src.tags()
                sun_az = float(tags.get("SUN_AZIMUTH", 0.0))
                sun_el = float(tags.get("SUN_ELEVATION", 45.0))
                sun = SunAngles(azimuth_deg=sun_az, elevation_deg=sun_el) if "SUN_AZIMUTH" in tags else None

        return GeoRaster(
            data=data,
            modality=modality,
            gsd_meters=gsd,
            sun_angles=sun,
            transform=transform,
            crs=crs,
            nodata_val=nodata,
        )

    @staticmethod
    def _read_pds4_image(
        image_path: Path,
        label_xml_path: Path,
        modality: SensorModality,
        gsd_fallback: float,
        allowed_dir: Optional[Path],
        sun_angles: Optional[SunAngles],
    ) -> GeoRaster:
        """Read a detached, row-major PDS4 2-D image described by an XML label."""
        safe_image = sanitize_path(image_path, allowed_dir=allowed_dir)
        safe_label = sanitize_path(label_xml_path, allowed_dir=allowed_dir)
        try:
            if LXML_AVAILABLE and lxml_ET is not None:
                parser = lxml_ET.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)
                root = lxml_ET.parse(str(safe_label), parser=parser).getroot()
            else:
                root = defused_ET.parse(str(safe_label), forbid_dtd=True, forbid_entities=True).getroot()
        except (DefusedXmlException, Exception) as exc:
            raise ValueError(f"Invalid or unsafe PDS4 XML label: {safe_label}") from exc

        def values(local_name: str) -> list[str]:
            return [
                node.text.strip()
                for node in root.iter()
                if node.text and node.tag.rsplit("}", 1)[-1] == local_name
            ]

        dimensions = [int(value) for value in values("elements")]
        if len(dimensions) < 2:
            raise ValueError(f"PDS4 label has no 2-D image dimensions: {safe_label}")
        height, width = dimensions[-2], dimensions[-1]
        offset_values = values("offset")
        byte_offset = int(float(offset_values[0])) if offset_values else 0
        data_type = (values("data_type") or ["MSB_INTEGER"])[0].upper()
        bits_values = values("bits")
        if bits_values:
            bits = int(bits_values[0])
        elif "BYTE" in data_type:
            bits = 8
        elif "WORD" in data_type:
            bits = 16
        elif "REAL" in data_type:
            bits = 32
        else:
            bits = 16

        if bits not in (8, 16, 32, 64):
            raise ValueError(f"Unsupported PDS4 sample width: {bits} bits")
        if "REAL" in data_type:
            dtype = np.dtype(">f4" if bits == 32 and "MSB" in data_type else "<f4")
        else:
            kind = "u" if "UNSIGNED" in data_type else "i"
            endian = ">" if "MSB" in data_type else "<"
            dtype = np.dtype(f"{endian}{kind}{bits // 8}")

        estimated_bytes = height * width * dtype.itemsize
        if height > MAX_RASTER_DIMENSION or width > MAX_RASTER_DIMENSION:
            raise ValueError(
                f"PDS4 raster dimensions ({width}x{height}) exceed security ceiling "
                f"({MAX_RASTER_DIMENSION}x{MAX_RASTER_DIMENSION}); use tiled ingestion."
            )
        if estimated_bytes > MAX_UNCOMPRESSED_BYTES:
            raise MemoryError(
                f"PDS4 raster buffer ({estimated_bytes / (1024**2):.1f} MB) exceeds safe threshold. "
                "Use PlanetaryTileProcessor for out-of-core processing."
            )

        with safe_image.open("rb") as stream:
            stream.seek(byte_offset)
            data = np.fromfile(stream, dtype=dtype, count=height * width)
        if data.size != height * width:
            raise ValueError(f"PDS4 image is truncated: expected {height * width} samples, found {data.size}")
        data = data.reshape((height, width)).astype(np.float32)

        parsed_sun, parsed_gsd, parsed_modality = PlanetaryRasterReader.parse_pds4_metadata(
            safe_label, allowed_dir=safe_label.parent
        )
        return GeoRaster(
            data=data,
            modality=parsed_modality if modality == SensorModality.SYNTHETIC else modality,
            gsd_meters=parsed_gsd if parsed_gsd > 0 else gsd_fallback,
            sun_angles=sun_angles or parsed_sun,
            crs="IAU2000:30100",
        )

    read_georaster = read_geotiff

    @staticmethod
    def parse_pds4_metadata(
        label_xml_path: Union[str, Path], allowed_dir: Optional[Path] = None
    ) -> Tuple[SunAngles, float, SensorModality]:
        """
        Parses PDS4 XML label for Chandrayaan-2/LRO products with hardened XXE protection:
        Ensures entity expansion is strictly disabled (`resolve_entities=False`).
        Extracts solar illumination angles, pixel resolution (GSD), and sensor modality.
        """
        safe_path = sanitize_path(label_xml_path, allowed_dir=allowed_dir)
        # Parse with strict XXE protection: entity expansion strictly disabled
        try:
            if LXML_AVAILABLE and lxml_ET is not None:
                parser = lxml_ET.XMLParser(
                    resolve_entities=False,
                    no_network=True,
                    dtd_validation=False,
                    load_dtd=False,
                )
                tree = lxml_ET.parse(str(safe_path), parser=parser)
                root = tree.getroot()
            else:
                tree = defused_ET.parse(str(safe_path), forbid_dtd=True, forbid_entities=True)
                root = tree.getroot()
        except (DefusedXmlException, Exception) as exc:
            raise ValueError(f"Invalid or unsafe PDS4 XML label: {safe_path}") from exc

        # Extract namespace if present
        ns = {"pds": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}

        # Default fallback values
        sun_az = 0.0
        sun_el = 45.0
        gsd = 1.0
        modality = SensorModality.SYNTHETIC

        def get_elem(tag_name: str):
            if ns:
                elem = root.find(f".//pds:{tag_name}", ns)
                if elem is not None:
                    return elem
            elem = root.find(f".//{tag_name}")
            if elem is not None:
                return elem
            # Mission-specific PDS4 dictionaries use additional namespaces
            # such as ISDA; match by local name without weakening XML safety.
            for candidate in root.iter():
                if candidate.tag.rsplit("}", 1)[-1] == tag_name and candidate.text:
                    return candidate
            return None

        # Search for solar geometry in PDS4 observation area
        az_node = get_elem("solar_azimuth_angle")
        if az_node is None:
            az_node = get_elem("sun_azimuth")
        el_node = get_elem("solar_elevation_angle")
        if el_node is None:
            el_node = get_elem("sun_elevation")
        gsd_node = get_elem("pixel_resolution")
        sensor_nodes = [
            candidate
            for candidate in root.iter()
            if candidate.tag.rsplit("}", 1)[-1] in {"instrument_id", "name"} and candidate.text
        ]

        if az_node is not None and az_node.text:
            sun_az = float(az_node.text)
        if el_node is not None and el_node.text:
            sun_el = float(el_node.text)
        if gsd_node is not None and gsd_node.text:
            gsd = float(gsd_node.text)

        sensor_text = " ".join(
            node.text for node in root.iter() if node.text and node.text.strip()
        ).upper()
        if "OHRC" in sensor_text:
            modality = SensorModality.OHRC
            gsd = gsd if gsd != 1.0 else 0.25
        elif "TMC" in sensor_text:
            modality = SensorModality.TMC2
            gsd = gsd if gsd != 1.0 else 5.0
        elif "IIRS" in sensor_text:
            modality = SensorModality.IIRS
            gsd = gsd if gsd != 1.0 else 80.0

        return SunAngles(azimuth_deg=sun_az, elevation_deg=sun_el), gsd, modality
