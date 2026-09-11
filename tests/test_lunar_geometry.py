import pytest

from lunar_core.data_io.lunar_geometry import (
    polygon_is_valid,
    spherical_polygon_area_km2,
    spherical_polygon_intersection_area_km2,
    unwrap_longitudes,
)


def test_longitudes_unwrap_across_dateline():
    values = unwrap_longitudes([179.0, -179.0, -178.0])
    assert values == [179.0, 181.0, 182.0]


def test_small_spherical_footprint_has_positive_area():
    area = spherical_polygon_area_km2([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    assert area > 0.0
    assert area < 100000.0


def test_invalid_footprint_is_rejected():
    assert not polygon_is_valid(None)
    assert not polygon_is_valid([(0.0, 0.0), (1.0, 1.0)])
    assert not polygon_is_valid([(0.0, 91.0), (1.0, 1.0), (2.0, 0.0)])
    assert polygon_is_valid([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])


def test_antimeridian_crossing_area_and_intersection():
    # Polygon spanning across 180° longitude
    poly1 = [(179.0, 0.0), (181.0, 0.0), (181.0, 1.0), (179.0, 1.0)]
    area1 = spherical_polygon_area_km2(poly1)
    assert area1 > 0.0

    poly2 = [(180.0, 0.0), (182.0, 0.0), (182.0, 1.0), (180.0, 1.0)]
    area2 = spherical_polygon_area_km2(poly2)
    assert area2 > 0.0

    # Intersection should be approximately half of poly1
    inter_area = spherical_polygon_intersection_area_km2(poly1, poly2)
    assert 0.0 < inter_area < area1
    assert abs(inter_area - 0.5 * area1) < 0.1 * area1


def test_polar_proximity_footprint():
    from lunar_core.data_io.lunar_geometry import validate_lunar_footprint
    south_pole_poly = [(0.0, -88.0), (90.0, -88.0), (180.0, -88.0), (270.0, -88.0)]
    valid, classification, details = validate_lunar_footprint(south_pole_poly, is_authoritative=True)
    assert valid is True
    assert classification == "VALIDATED_GEOSPATIAL"
    assert "near lunar pole" in details


def test_spherical_haversine_distance():
    from lunar_core.data_io.lunar_geometry import spherical_distance_km
    # Distance from pole to equator along meridian is ~ 1/4 circumference = 1737.4 * pi / 2 = ~2729 km
    dist = spherical_distance_km(0.0, 0.0, 90.0, 0.0)
    assert 2720.0 < dist < 2740.0

