"""Canonical metadata model for mission products and catalog manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class MissionProduct:
    """A discovered planetary product, without loading its pixel data."""

    mission: Optional[str]
    instrument: Optional[str]
    product_id: str
    image_path: Path
    label_path: Optional[Path] = None
    file_format: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    band_count: Optional[int] = None
    dtype: Optional[str] = None
    gsd_m: Optional[float] = None
    acquisition_time: Optional[str] = None
    center_lat_deg: Optional[float] = None
    center_lon_deg: Optional[float] = None
    crs: Optional[str] = None
    projection: Optional[str] = None
    sun_azimuth_deg: Optional[float] = None
    sun_elevation_deg: Optional[float] = None
    incidence_angle_deg: Optional[float] = None
    processing_level: Optional[str] = None
    product_type: Optional[str] = None
    metadata_source: Optional[str] = None
    status: str = "validated"
    validation_message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/CSV-friendly representation with string paths."""
        record = asdict(self)
        record["image_path"] = str(self.image_path)
        record["label_path"] = str(self.label_path) if self.label_path else None
        return record
