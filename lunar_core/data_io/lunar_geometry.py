"""Geometry helpers for lunar geographic footprints.

The lunar body is modeled as a unit sphere for metadata-level geometry. This
module provides spherical polygon area and longitude unwrapping. Polygon
intersection remains a conservative equirectangular prefilter unless a product
supplies an authoritative projected footprint; callers must not treat the
prefilter as scientific overlap truth.
"""

from __future__ import annotations

import math
from typing import Iterable


def unwrap_longitudes(longitudes: Iterable[float]) -> list[float]:
    values = [float(value) for value in longitudes]
    if not values:
        return []
    unwrapped = [values[0]]
    for value in values[1:]:
        candidate = value
        while candidate - unwrapped[-1] > 180.0:
            candidate -= 360.0
        while candidate - unwrapped[-1] < -180.0:
            candidate += 360.0
        unwrapped.append(candidate)
    return unwrapped


def _unit_vector(lon_deg: float, lat_deg: float) -> tuple[float, float, float]:
    lon = math.radians(lon_deg)
    lat = math.radians(lat_deg)
    return math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)


def spherical_polygon_area_km2(polygon: list[tuple[float, float]], radius_km: float = 1737.4) -> float:
    """Return the smaller spherical polygon area in square kilometres."""
    if len(polygon) < 3:
        return 0.0
    lons = unwrap_longitudes(point[0] for point in polygon)
    normalized = [(lons[index], polygon[index][1]) for index in range(len(polygon))]
    vectors = [_unit_vector(lon, lat) for lon, lat in normalized]
    area_sr = 0.0
    anchor = vectors[0]
    for index in range(1, len(vectors) - 1):
        first = vectors[index]
        second = vectors[index + 1]
        cross_norm = math.sqrt(sum(value * value for value in (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        )))
        dot_sum = sum(anchor[pos] * first[pos] for pos in range(3))
        dot_sum += sum(first[pos] * second[pos] for pos in range(3))
        dot_sum += sum(second[pos] * anchor[pos] for pos in range(3))
        dot_sum += 1.0
        area_sr += 2.0 * math.atan2(cross_norm, max(dot_sum, 1e-15))
    return abs(area_sr) * radius_km * radius_km


def polygon_is_valid(polygon: list[tuple[float, float]] | None) -> bool:
    if polygon is None or len(polygon) < 3:
        return False
    return all(math.isfinite(float(lon)) and math.isfinite(float(lat)) and -90.0 <= float(lat) <= 90.0 for lon, lat in polygon)
