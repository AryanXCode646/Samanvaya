import pytest

from lunar_core.data_io.lunar_geometry import polygon_is_valid, spherical_polygon_area_km2, unwrap_longitudes


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
