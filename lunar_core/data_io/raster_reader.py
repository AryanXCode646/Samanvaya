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
MAX_RASTER_DIMENSION = 30000          # 30,000 x 30,000 max single raster dimension
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
            return root.find(f".//{tag_name}")

        # Search for solar geometry in PDS4 observation area
        az_node = get_elem("solar_azimuth_angle")
        el_node = get_elem("solar_elevation_angle")
        gsd_node = get_elem("pixel_resolution")
        sensor_node = get_elem("instrument_id")

        if az_node is not None and az_node.text:
            sun_az = float(az_node.text)
        if el_node is not None and el_node.text:
            sun_el = float(el_node.text)
        if gsd_node is not None and gsd_node.text:
            gsd = float(gsd_node.text)

        if sensor_node is not None and sensor_node.text:
            s_name = sensor_node.text.upper()
            if "OHRC" in s_name:
                modality = SensorModality.OHRC
                gsd = gsd if gsd != 1.0 else 0.25
            elif "TMC" in s_name:
                modality = SensorModality.TMC2
                gsd = gsd if gsd != 1.0 else 5.0
            elif "IIRS" in s_name:
                modality = SensorModality.IIRS
                gsd = gsd if gsd != 1.0 else 80.0

        return SunAngles(azimuth_deg=sun_az, elevation_deg=sun_el), gsd, modality
