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


# IAU/IAG 2015 Lunar Reference System Constants
LUNAR_RADIUS_KM = 1737.4
BODY_FRAME = "IAU_2015_MOON"
LATITUDE_CONVENTION = "PLANETOCENTRIC"
LONGITUDE_CONVENTION = "POSITIVE_EAST"

LUNAR_GEOMETRY_FRAME = {
    "body": "MOON",
    "datum": "IAU_2015_SPHERE",
    "radius_km": LUNAR_RADIUS_KM,
    "latitude_convention": LATITUDE_CONVENTION,
    "longitude_convention": LONGITUDE_CONVENTION,
    "latitude_bounds_deg": [-90.0, 90.0],
    "longitude_bounds_deg": [-180.0, 360.0],
    "units": {"distance": "km", "area": "km2", "resolution": "meters/pixel"},
}


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


def spherical_distance_km(
    lat1_deg: float,
    lon1_deg: float,
    lat2_deg: float,
    lon2_deg: float,
    radius_km: float = LUNAR_RADIUS_KM,
) -> float:
    """Great-circle distance between two points on the lunar sphere (Haversine formula)."""
    phi1, phi2 = math.radians(lat1_deg), math.radians(lat2_deg)
    delta_phi = math.radians(lat2_deg - lat1_deg)
    delta_lambda = math.radians(lon2_deg - lon1_deg)
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(min(1.0, max(0.0, a))), math.sqrt(min(1.0, max(0.0, 1.0 - a))))
    return radius_km * c


def spherical_polygon_area_km2(polygon: list[tuple[float, float]], radius_km: float = LUNAR_RADIUS_KM) -> float:
    """Return the smaller spherical polygon area in square kilometres using the Oosterom & Strackee (1983) formula."""
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
        cross_bc = (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        )
        det = abs(anchor[0] * cross_bc[0] + anchor[1] * cross_bc[1] + anchor[2] * cross_bc[2])
        dot_sum = (
            1.0
            + sum(anchor[pos] * first[pos] for pos in range(3))
            + sum(first[pos] * second[pos] for pos in range(3))
            + sum(second[pos] * anchor[pos] for pos in range(3))
        )
        area_sr += 2.0 * math.atan2(det, max(dot_sum, 1e-15))
    return abs(area_sr) * radius_km * radius_km


def _clip_polygon_unwrapped(
    subject: list[tuple[float, float]],
    edge_start: tuple[float, float],
    edge_end: tuple[float, float],
) -> list[tuple[float, float]]:
    """Clip a 2-D polygon against one directed edge in unwrapped coordinate space."""
    if not subject:
        return []

    def cross(first: tuple[float, float], second: tuple[float, float], point: tuple[float, float]) -> float:
        return (second[0] - first[0]) * (point[1] - first[1]) - (second[1] - first[1]) * (point[0] - first[0])

    output: list[tuple[float, float]] = []
    previous = subject[-1]
    previous_inside = cross(edge_start, edge_end, previous) >= 0
    for current in subject:
        current_inside = cross(edge_start, edge_end, current) >= 0
        if current_inside != previous_inside:
            denominator = cross(edge_start, edge_end, current) - cross(edge_start, edge_end, previous)
            if abs(denominator) > 1e-12:
                ratio = -cross(edge_start, edge_end, previous) / denominator
                output.append(
                    (previous[0] + ratio * (current[0] - previous[0]), previous[1] + ratio * (current[1] - previous[1]))
                )
        if current_inside:
            output.append(current)
        previous, previous_inside = current, current_inside
    return output


def spherical_polygon_intersection_area_km2(
    first: list[tuple[float, float]],
    second: list[tuple[float, float]],
    radius_km: float = LUNAR_RADIUS_KM,
) -> float:
    """Compute intersection area between two lunar footprint polygons in km2."""
    if not polygon_is_valid(first) or not polygon_is_valid(second):
        return 0.0

    # Align longitudes relative to the center of the first polygon to prevent antimeridian discontinuity
    mean_lon1 = sum(pt[0] for pt in first) / len(first)
    
    def _recenter(poly: list[tuple[float, float]], ref_lon: float) -> list[tuple[float, float]]:
        out = []
        for lon, lat in poly:
            d = lon - ref_lon
            while d > 180.0:
                d -= 360.0
            while d < -180.0:
                d += 360.0
            out.append((ref_lon + d, lat))
        return out

    poly1 = _recenter(first, mean_lon1)
    poly2 = _recenter(second, mean_lon1)

    clipped = list(poly1)
    for index, edge_start in enumerate(poly2):
        clipped = _clip_polygon_unwrapped(clipped, edge_start, poly2[(index + 1) % len(poly2)])
        if not clipped:
            return 0.0

    return spherical_polygon_area_km2(clipped, radius_km=radius_km)


def polygon_is_valid(polygon: list[tuple[float, float]] | None) -> bool:
    """Check if polygon vertices have valid coordinates within planetary physical bounds."""
    if polygon is None or len(polygon) < 3:
        return False
    return all(
        math.isfinite(float(lon))
        and math.isfinite(float(lat))
        and -90.0 <= float(lat) <= 90.0
        and -360.0 <= float(lon) <= 720.0
        for lon, lat in polygon
    )


def validate_lunar_footprint(
    polygon: list[tuple[float, float]] | None,
    is_authoritative: bool = False,
) -> tuple[bool, str, str]:
    """Validate and classify a footprint polygon.
    
    Returns:
        (is_valid, classification, details)
        where classification is one of:
        - "VALIDATED_GEOSPATIAL" (authoritative boundary on lunar sphere)
        - "APPROXIMATE_GEOGRAPHIC" (bounding box or uncalibrated corners)
        - "INVALID_GEOMETRY" (non-finite or out-of-bounds coords)
        - "GEOMETRY_UNAVAILABLE" (missing footprint)
    """
    if polygon is None or len(polygon) == 0:
        return False, "GEOMETRY_UNAVAILABLE", "Footprint coordinates are absent."
    if not polygon_is_valid(polygon):
        return False, "INVALID_GEOMETRY", "Polygon contains non-finite or out-of-bounds coordinates."
    
    # Check polar proximity
    near_pole = any(abs(float(lat)) > 85.0 for _, lat in polygon)
    pole_note = " (near lunar pole: longitude convergence active)" if near_pole else ""
    
    if is_authoritative:
        return True, "VALIDATED_GEOSPATIAL", f"Authoritative projected lunar footprint{pole_note}."
    return True, "APPROXIMATE_GEOGRAPHIC", f"Approximate geographic footprint (bounding corners){pole_note}."

