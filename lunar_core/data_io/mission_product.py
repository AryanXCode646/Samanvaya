"""Canonical metadata model for mission products and catalog manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from enum import Enum
from typing import Any, Optional


class ProductStatus(str, Enum):
    """Lifecycle state for discovered mission products."""

    DISCOVERED = "discovered"
    PARSED = "parsed"
    VALIDATED = "validated"
    PARTIAL = "partial"
    INVALID = "invalid"
    UNSUPPORTED = "unsupported"


@dataclass
class MissionProduct:
    """A discovered planetary product without loading raster pixels into memory."""

    mission: Optional[str] = None
    instrument: Optional[str] = None
    product_id: Optional[str] = None
    image_path: Optional[Path] = None
    label_path: Optional[Path] = None
    file_format: Optional[str] = None
    format: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    band_count: Optional[int] = None
    dtype: Optional[str] = None
    gsd_m: Optional[float] = None
    gsd_source: Optional[str] = None
    wavelength_min_um: Optional[float] = None
    wavelength_max_um: Optional[float] = None
    acquisition_time: Optional[str] = None
    center_lat_deg: Optional[float] = None
    center_lon_deg: Optional[float] = None
    footprint: Optional[list[tuple[float, float]]] = None
    crs: Optional[str] = None
    projection: Optional[str] = None
    spacecraft: Optional[str] = None
    spacecraft_name: Optional[str] = None
    sun_azimuth_deg: Optional[float] = None
    sun_elevation_deg: Optional[float] = None
    sun_geometry_source: Optional[str] = None
    incidence_angle_deg: Optional[float] = None
    emission_angle_deg: Optional[float] = None
    phase_angle_deg: Optional[float] = None
    fill_value: Optional[float] = None
    no_data_value: Optional[float] = None
    processing_level: Optional[str] = None
    product_type: Optional[str] = None
    metadata_source: Optional[str] = None
    identification_method: Optional[str] = None
    status: ProductStatus | str = ProductStatus.DISCOVERED
    validation_status: Optional[str] = None
    validation_message: Optional[str] = None

    def __post_init__(self) -> None:
        if self.file_format is None and self.format is not None:
            self.file_format = self.format
        if self.format is None and self.file_format is not None:
            self.format = self.file_format
        if self.spacecraft_name is None and self.spacecraft is not None:
            self.spacecraft_name = self.spacecraft
        if self.spacecraft is None and self.spacecraft_name is not None:
            self.spacecraft = self.spacecraft_name
        if self.validation_status is None:
            self.validation_status = self.status

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/CSV-friendly representation with string paths."""
        record = asdict(self)
        if self.image_path is not None:
            record["image_path"] = str(self.image_path)
        else:
            record["image_path"] = None
        if self.label_path is not None:
            record["label_path"] = str(self.label_path)
        else:
            record["label_path"] = None
        if record.get("footprint") is not None:
            record["footprint"] = [list(point) for point in self.footprint]
        return record
