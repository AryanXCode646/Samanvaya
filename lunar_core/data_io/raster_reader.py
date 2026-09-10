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
            from lunar_core.data_io.product_identity import resolve_product_label

            resolution = resolve_product_label(path)
            if not resolution.resolved or resolution.path is None:
                raise FileNotFoundError(
                    resolution.message or f"PDS4 XML label not found for detached image: {path}"
                )
            label_path = resolution.path
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
                try:
                    sun_az = float(tags["SUN_AZIMUTH"])
                    sun_el = float(tags["SUN_ELEVATION"])
                    sun = SunAngles(azimuth_deg=sun_az, elevation_deg=sun_el)
                except (KeyError, TypeError, ValueError):
                    sun = None

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

    @staticmethod
    def open_pds4_memmap(
        image_path: Union[str, Path],
        label_xml_path: Union[str, Path],
        allowed_dir: Optional[Path] = None,
    ) -> np.memmap:
        """Open a detached 2-D PDS4 image lazily for windowed processing."""
        safe_image = sanitize_path(image_path, allowed_dir=allowed_dir)
        safe_label = sanitize_path(label_xml_path, allowed_dir=allowed_dir)
        try:
            root = defused_ET.parse(safe_label, forbid_dtd=True, forbid_entities=True).getroot()
        except Exception as exc:
            raise ValueError(f"Invalid or unsafe PDS4 XML label: {safe_label}") from exc

        def values(name: str) -> list[str]:
            return [
                node.text.strip()
                for node in root.iter()
                if node.text and node.tag.rsplit("}", 1)[-1].lower() == name.lower()
            ]

        dimensions = [int(value) for value in values("elements")]
        if len(dimensions) < 2:
            raise ValueError(f"PDS4 label has no 2-D image dimensions: {safe_label}")
        height, width = dimensions[-2], dimensions[-1]
        offsets = values("offset")
        byte_offset = int(float(offsets[0])) if offsets else 0
        data_type = (values("data_type") or ["MSB_INTEGER"])[0].upper()
        bits_values = values("bits")
        bits = int(bits_values[0]) if bits_values else (8 if "BYTE" in data_type else 32 if "REAL" in data_type else 16)
        if bits not in (8, 16, 32, 64):
            raise ValueError(f"Unsupported PDS4 sample width: {bits} bits")
        if "REAL" in data_type:
            dtype = np.dtype(">f4" if bits == 32 and "MSB" in data_type else "<f4")
        else:
            kind = "u" if "UNSIGNED" in data_type else "i"
            endian = ">" if "MSB" in data_type else "<"
            dtype = np.dtype(f"{endian}{kind}{bits // 8}")
        expected_size = byte_offset + height * width * dtype.itemsize
        if safe_image.stat().st_size < expected_size:
            raise ValueError(f"PDS4 image is truncated: expected {expected_size} bytes, found {safe_image.stat().st_size}")
        return np.memmap(safe_image, dtype=dtype, mode="r", offset=byte_offset, shape=(height, width), order="C")

    @staticmethod
    def open_pds4_spectral_memmap(
        image_path: Union[str, Path],
        label_xml_path: Union[str, Path],
        allowed_dir: Optional[Path] = None,
    ) -> tuple[np.memmap, tuple[int, int, int]]:
        """Open a PDS4 Array_3D_Spectrum as a bands-first lazy array."""
        safe_image = sanitize_path(image_path, allowed_dir=allowed_dir)
        safe_label = sanitize_path(label_xml_path, allowed_dir=allowed_dir)
        try:
            root = defused_ET.parse(safe_label, forbid_dtd=True, forbid_entities=True).getroot()
        except Exception as exc:
            raise ValueError(f"Invalid or unsafe PDS4 XML label: {safe_label}") from exc

        array = next(
            (node for node in root.iter() if node.tag.rsplit("}", 1)[-1].lower() == "array_3d_spectrum"),
            None,
        )
        if array is None:
            raise ValueError(f"PDS4 label has no Array_3D_Spectrum: {safe_label}")

        axes: list[tuple[int, str, int]] = []
        for axis in array.iter():
            if axis.tag.rsplit("}", 1)[-1].lower() != "axis_array":
                continue
            name = next(
                (child.text.strip().upper() for child in axis if child.text and child.tag.rsplit("}", 1)[-1].lower() == "axis_name"),
                "",
            )
            elements = next(
                (int(child.text.strip()) for child in axis if child.text and child.tag.rsplit("}", 1)[-1].lower() == "elements"),
                0,
            )
            sequence = next(
                (int(child.text.strip()) for child in axis if child.text and child.tag.rsplit("}", 1)[-1].lower() == "sequence_number"),
                len(axes) + 1,
            )
            if elements <= 0:
                raise ValueError(f"Invalid PDS4 spectral axis {name!r}: {elements}")
            axes.append((sequence, name, elements))
        if len(axes) != 3:
            raise ValueError(f"Expected three PDS4 spectral axes, found {len(axes)}")

        ordered = sorted(axes)
        shape = tuple(axis[2] for axis in ordered)
        names = [axis[1] for axis in ordered]
        if names[0] != "BAND" or names[1:] != ["LINE", "SAMPLE"]:
            raise ValueError(f"Unsupported PDS4 spectral axis order: {names}")
        element = next(
            (node for node in array.iter() if node.tag.rsplit("}", 1)[-1].lower() == "element_array"),
            None,
        )
        data_type = next(
            (child.text.strip().upper() for child in element or [] if child.text and child.tag.rsplit("}", 1)[-1].lower() == "data_type"),
            "MSB_INTEGER",
        )
        bits = 32 if "REAL" in data_type else 16
        if "IEEE754L SBSINGLE" in data_type.replace(" ", "") or "LSBSINGLE" in data_type:
            dtype = np.dtype("<f4")
        elif "REAL" in data_type:
            dtype = np.dtype(">f4" if "MSB" in data_type else "<f4")
        else:
            raise ValueError(f"Unsupported PDS4 spectral sample type: {data_type}")
        offset_node = next(
            (node for node in array if node.tag.rsplit("}", 1)[-1].lower() == "offset" and node.text),
            None,
        )
        offset = int(float(offset_node.text)) if offset_node is not None else 0
        expected_size = offset + int(np.prod(shape)) * dtype.itemsize
        if safe_image.stat().st_size < expected_size:
            raise ValueError(f"PDS4 spectral cube is truncated: expected {expected_size} bytes, found {safe_image.stat().st_size}")
        return np.memmap(safe_image, dtype=dtype, mode="r", offset=offset, shape=shape, order="C"), shape

    read_georaster = read_geotiff

    @staticmethod
    def parse_pds4_metadata(
        label_xml_path: Union[str, Path], allowed_dir: Optional[Path] = None
    ) -> Tuple[Optional[SunAngles], float, SensorModality]:
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

        def _local_name(tag: str) -> str:
            return tag.rsplit("}", 1)[-1]

        def _text_from_candidates(*names: str) -> list[str]:
            values: list[str] = []
            for tag_name in names:
                for candidate in root.iter():
                    if _local_name(candidate.tag).lower() == tag_name.lower() and candidate.text:
                        values.append(candidate.text.strip())
            return values

        def _coerce_float(raw: str) -> Optional[float]:
            try:
                value = float(raw)
                return value if np.isfinite(value) else None
            except (TypeError, ValueError):
                import re
                match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
                if match is None:
                    return None
                try:
                    value = float(match.group(0))
                    return value if np.isfinite(value) else None
                except ValueError:
                    return None

        def _distance_in_meters(element: Any) -> Optional[float]:
            if element is None or not element.text:
                return None
            value = _coerce_float(element.text)
            if value is None or value <= 0:
                return None
            unit = (element.attrib.get("unit", "m") or "m").strip().lower()
            factors = {
                "m": 1.0,
                "meter": 1.0,
                "meters": 1.0,
                "metre": 1.0,
                "metres": 1.0,
                "cm": 1e-2,
                "centimeter": 1e-2,
                "centimeters": 1e-2,
                "mm": 1e-3,
                "millimeter": 1e-3,
                "millimeters": 1e-3,
                "um": 1e-6,
                "micrometer": 1e-6,
                "micrometers": 1e-6,
                "km": 1e3,
                "kilometer": 1e3,
                "kilometers": 1e3,
            }
            factor = factors.get(unit)
            return value * factor if factor is not None else None

        def get_elem(*tag_names: str):
            visited: set[str] = set()
            for tag_name in tag_names:
                query = tag_name.lower()
                if query in visited:
                    continue
                visited.add(query)
                if ns:
                    elem = root.find(f".//pds:{tag_name}", ns)
                    if elem is not None:
                        return elem
                elem = root.find(f".//{tag_name}")
                if elem is not None:
                    return elem
                for candidate in root.iter():
                    if _local_name(candidate.tag).lower() == query and candidate.text:
                        return candidate
            return None

        # Default fallback values
        sun_az: Optional[float] = None
        sun_el: Optional[float] = None
        gsd = 1.0
        modality = SensorModality.SYNTHETIC

        az_node = get_elem("solar_azimuth_angle", "sun_azimuth", "solar_azimuth", "sun_azimuth_deg")
        el_node = get_elem("solar_elevation_angle", "sun_elevation", "solar_elevation", "sun_elevation_deg")
        gsd_node = get_elem("pixel_resolution", "ground_sample_distance", "gsd", "resolution")

        if az_node is not None and az_node.text:
            parsed = _coerce_float(az_node.text)
            if parsed is not None:
                sun_az = parsed
        if el_node is not None and el_node.text:
            parsed = _coerce_float(el_node.text)
            if parsed is not None:
                sun_el = parsed
        if gsd_node is not None and gsd_node.text:
            parsed = _distance_in_meters(gsd_node)
            if parsed is not None:
                gsd = parsed

        if gsd == 1.0:
            for candidate_text in _text_from_candidates("pixel_resolution", "ground_sample_distance", "gsd"):
                parsed = _coerce_float(candidate_text)
                if parsed is not None:
                    if parsed > 0:
                        gsd = parsed
                        break

        from lunar_core.data_io.product_identity import identify_mission_instrument

        _mission, instrument, _method = identify_mission_instrument(safe_path, root)
        instrument_to_modality = {
            "OHRC": (SensorModality.OHRC, 0.25),
            "TMC-2": (SensorModality.TMC2, 5.0),
            "IIRS": (SensorModality.IIRS, 80.0),
            "HYSI": (SensorModality.HYSI, 80.0),
            "NAC": (SensorModality.LRO_NAC, 0.5),
        }
        if instrument in instrument_to_modality:
            modality, default_gsd = instrument_to_modality[instrument]
            if gsd == 1.0:
                gsd = default_gsd

        sun = SunAngles(azimuth_deg=sun_az, elevation_deg=sun_el) if sun_az is not None and sun_el is not None else None
        return sun, gsd, modality
